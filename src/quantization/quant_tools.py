import torch
import torch.nn.functional as F

import torchvision

# Quantized Tensor ====================================================================


class QuantizedTensor(object):
    """
    Base class for Quantized Tensor
    Contains info tensor, scale, zero_point
    """

    def __init__(self, data, metadata: dict = {
        "scale": None, "zero_point": None, "bitwidth": None
    }, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self._metadata = metadata

    # Getters and setters ======================
    def _get_param(self, value: str):
        res = self._metadata.get(value)
        if res == None:
            raise AttributeError(f"{value} is not setted")
        return res

    # Getters
    def get_scale(self):
        return self._get_param("scale")

    def get_zero_point(self):
        return self._get_param("zero_point")

    def get_bitwidth(self):
        return self._get_param("bitwidth")

    # Setters
    def set_scale(self, value):
        self._metadata["scale"] = value

    def set_zero_point(self, value):
        self._metadata["zero_point"] = value

    def set_bitwidth(self, value):
        self._metadata["bitwidth"] = value

    # Getters and setters end ======================

    def to_tensor(self):
        return torch.Tensor(self._t)

    def to(self, device):
        result = QuantizedTensor(
            data=self._t.to(device)
        )
        return result

    def __repr__(self):
        return f"QT({self._t})"

    # Add here checking for metadatas also in kwargs objects
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
        args = [a._t if hasattr(a, '_t') else a for a in args]
        assert len(metadatas) > 0, f"heh"
        ret = func(*args, **kwargs)
        return QuantizedTensor(ret, metadata=metadatas[0])


# Quantization functions ====================================================================
# Uniform quantization ==========================================
def __quantize_tensor(tensor: torch.Tensor, scale: int, zero_point: int, device: str = 'cpu') -> QuantizedTensor:
    res = QuantizedTensor(tensor).to(device)
    res.set_scale(scale)
    res.set_zero_point(zero_point)

    # FIXME: Was cuda
    return torch.round(res/scale).to(device) - torch.tensor([zero_point]).to(device)

# Calculations ======================================================


def calc_scale(tensor: torch.Tensor, bitwidth: int = 8, symmetric: bool = False) -> int:
    """
    Calc scale for assymetric quantization
    """
    if symmetric:
        b = tensor.abs().max()
        a = -b
    else:
        b = tensor.max()
        a = tensor.min()

    scale = (b-a)/(2**bitwidth-1)
    print(f"Scale: {scale}")

    return scale


def quantize(tensor: torch.Tensor, bitwidth: int = 8, train: bool = True, symmetric: bool = False, device='cpu') -> QuantizedTensor:
    """
    Quantize tensor to custom bitwidth
    Using Asymmetric quantization
    """

    a = tensor.min()
    b = tensor.max()

    a_q = -2**(bitwidth-1)
    b_q = 2**(bitwidth-1)-1

    scale = (b-a)/(b_q-a_q)

    zero_point = torch.round((a*b_q - b*a_q) / (b-a))

    res = __quantize_tensor(tensor, scale, zero_point, device=device)

    if not train:
        res = res.to_tensor()

    res = torch.clip(res, a_q, b_q)
    return res


def dequantize(tensor: QuantizedTensor) -> torch.Tensor:
    """
    Dequantize tensor
    """
    scale = tensor.get_scale()
    zero_point = tensor.get_zero_point()

    res = scale*(tensor+zero_point)
    res = res.to_tensor()
    return res


# Quantization Aware Training ====================================================================
class FakeQuantOp(torch.autograd.Function):
    """
    Implements module for Quantization Aware Training, 
    using Straight Through Estimator (STE) for backward pass 
    """
    @staticmethod
    def forward(ctx, x, num_bits=8, device='cpu', min_val=None, max_val=None):
        x = quantize(x, bitwidth=num_bits, device=device)
        x = dequantize(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None


def quant_forward_train(model: torch.nn.Module, features_batch: torch.Tensor, bit_width: int = 8, device: str = 'cuda'):
    """
    Implements forward pass using Quantization aware training
    """
    for name, module in model.named_children():
        # Sequential or Bottleneck
        if isinstance(module, torch.nn.Sequential) or \
           isinstance(module, torchvision.models.resnet.Bottleneck):
            features_batch = quant_forward_train(
                module, features_batch, bit_width, device)

        # BatchNorm
        elif isinstance(module, torch.nn.BatchNorm2d):
            # Do nothing with batchnorm
            features_batch = module(features_batch)

        # Conv2d
        elif isinstance(module, torch.nn.Conv2d):
            # Channelwise quantization (now not implement due to performance issues)
            # Quantize weights for each filter independently
            weights = module.weight.data
            weights = weights.to(device)

            # FIXME: This is bottleneck (channel_wise convolution)
            # for i in range(len(weights)):
            #     for j in range(len(weights[i])):
            #         weights[i][j] = FakeQuantOp.apply(weights[i][j], bit_width)

            # Maybe i should implement here channel-wise quantization for convolution
            weights = FakeQuantOp.apply(weights, bit_width, device)

            module.weight.data = weights
            features_batch = module(features_batch)

        # Linear
        elif isinstance(module, torch.nn.Linear):
            features_batch = torch.flatten(features_batch, start_dim=1)

            # Get weights
            weights = module.weight.data
            weights = weights.to(device)

            # Apply fake quantization
            new_weights = FakeQuantOp.apply(weights, bit_width, device)

            # Update weights
            module.weight.data = new_weights

            # Apply layer
            features_batch = module(features_batch)

    return features_batch


def quant_forward_eval(model: torch.nn.Module, features_batch: torch.Tensor, bit_width: int = 8, device: str = 'cpu'):
    """
    Forward pass in eval mode using quantized weights
    Device strictly cpu
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Sequential) or \
           isinstance(module, torchvision.models.resnet.Bottleneck):
            features_batch = quant_forward_train(
                module, features_batch, bit_width, device)

        elif isinstance(module, torch.nn.BatchNorm2d):
            features_batch = module(features_batch)

        elif isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data
            quantized_weights = quantize(weights, bit_width, train=False)
            features_batch = F.conv2d(
                input=features_batch,
                weight=quantized_weights
            )

        elif isinstance(module, torch.nn.Linear):
            # Flatten features batch
            features_batch = torch.flatten(features_batch, start_dim=1)

            # Get weights
            weights = module.weight.data

            # Quantize weights
            quantized_weights = quantize(weights, bit_width, train=False)

            # Apply linear operation using quantized weights
            features_batch = F.linear(
                input=features_batch,
                weight=quantized_weights
            )

    return features_batch
