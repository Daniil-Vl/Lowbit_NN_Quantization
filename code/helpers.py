import os
import time
import torch

from quant_tools import quant_forward_train, quant_forward_eval


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        print(
            f"Time consumption of func {func.__name__} = {time.time() - start_time:.2f} secs")

    return wrapper


# TODO: Implement learning rate scheduler support
@time_it
def train_model(model, dataloader, loss_fn, optimizer, epochs, device='cpu', qat: bool = False, bit_width: int = 8) -> list:
    """
    Implements training loop for model
    Returns history of loss and accuracy during epochs
    """
    if qat:
        print(f"Bitwidth for QAT: {bit_width}")

    history = {'loss': [], 'accuracy': []}

    steps_per_epoch = len(dataloader)
    model.train()
    model.to(device)

    n_total = len(dataloader.dataset)

    for epoch in range(epochs):
        cum_loss = 0
        n_correct = 0
        for (features, labels) in dataloader:

            # Move batches to device (cuda if available)
            features = features.to(device)
            labels = labels.to(device)

            # Make predictions and calculate loss
            # If qat == True, train model, using Quantization aware training
            if qat:
                predictions = quant_forward_train(
                    model, features, bit_width, device=device)
            else:
                predictions = model(features)
            loss = loss_fn(predictions, labels)
            cum_loss += loss.item()

            # Check accuracy on train dataset
            class_predictions = torch.argmax(input=predictions, dim=1)
            n_correct += (labels == class_predictions).sum().item()

            # Calculate gradient and make gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save loss
        history['loss'].append(cum_loss / steps_per_epoch)

        # Save accuracy
        history['accuracy'].append(n_correct / n_total)

        # Loss tracking between different epochs
        print(f"Epoch {epoch+1}")
        print("-------------------------------")
        print(f"Average loss = {cum_loss / steps_per_epoch:.5f}")

        # Accuracy tracking
        print(f"Average accuracy = {n_correct / n_total:.5f}")
        print("-------------------------------")


@time_it
def test_model(model: torch.nn.Module, dataloader, qat: bool = False, bitwidth: int = 8) -> None:
    """
    This function implements accuracy measuring on passed dataloader
    """
    model.to('cpu')
    model.eval()

    n_correct = 0
    n_total = len(dataloader.dataset)

    with torch.no_grad():
        for (images, labels) in dataloader:
            if qat:
                logits = quant_forward_eval(
                    model=model,
                    features_batch=images,
                    bit_width=bitwidth
                )
            else:
                logits = model(images)
            predictions = torch.argmax(input=logits, dim=1)
            n_correct += (labels == predictions).sum().item()

    print(f"Accuracy = {n_correct / n_total*100:.2f}%")


# TODO: Rewrite to save more info
def save_model(model, filename):
    torch.save(model.state_dict(), "params/" + filename)


def model_size(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size/1e3)
    os.remove('temp.p')
    return size
