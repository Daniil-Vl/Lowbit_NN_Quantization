"""
Train and save best models parameters
Attention: this script alawys will rewrite model
Filename: model_from_train_py.pt

Command usage: `python train.py [options]`
Example `python train.py --epochs=10`
Options syntax: --epochs=<value>
Options:
    1) --epochs - number of epochs for training
    2) --kernel_size - choose pane
"""
import sys
import argparse

import numpy as np

import torch
import torchvision
from torchsummary import summary

from src.models import ModelM3, ModelM5, ModelM7, WideModelM5, ResNet, ModelV6
from src.helpers import train_model, test_model, save_model
from src.helpers.transforms import RandomRotation


def validate_params(params: argparse.Namespace) -> None:
    for name, value in params._get_kwargs():
        if name == "kernel_size":
            if value not in (3, 5, 7):
                raise argparse.ArgumentError(
                    message="Kernel size should be 3 or 5 or 7!!!!")
        elif name == "bitwidth":
            if value not in (1, 2, 4, 8):
                raise argparse.ArgumentError(
                    message="Bitwidth for quantization allowed only with value 1, 2, 4, 8")


# Main function ================================================================================
def run(
        epochs: int = 10,
        kernel_size: int = 5,
        seed: int = 42,
        qat: bool = False,
        bitwidth: int = 8,
        filename: str = None,
        model_name: str = None,) -> None:
    # random number generator seed ------------------------------------------------#
    SEED = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # Kernel size of model --------------------------------------------------------#
    KERNEL_SIZE = kernel_size

    # Number of epochs ------------------------------------------------------------#
    NUM_EPOCHS = epochs

    # Batch size ------------------------------------------------------------#
    BATCH_SIZE = 128

    # Learning rate ------------------------------------------------------------#
    LEARNING_RATE = 1e-3

    # Logging dirs ----------------------------------------------------------------#
    # TODO: Implement logging

    # Enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        exit(0)

    # Data augmentation methods ---------------------------------------------------#
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # Data loaders -----------------------------------------------------------------#
    train_dataset = torchvision.datasets.EMNIST(
        root="data",
        split="mnist",
        train=True,
        download=False,
        transform=transformation,
    )

    test_dataset = torchvision.datasets.EMNIST(
        root="data",
        split="mnist",
        train=False,
        download=False,
        transform=transformation,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model selection -------------------------------------------------------------#
    if model_name == "WideModelM5":
        model = WideModelM5().to(device)
    elif model_name.lower() == "resnet":
        model = ResNet().to(device)
    elif model_name.lower() == "modelv6":
        model = ModelV6().to(device)
    elif (KERNEL_SIZE == 3):
        model = ModelM3().to(device)
    elif (KERNEL_SIZE == 5):
        model = ModelM5().to(device)
    elif (KERNEL_SIZE == 7):
        model = ModelM7().to(device)

    summary(model, (1, 28, 28))

    # Hyperparameter selection ----------------------------------------------------#
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Parameters for stopping -----------------------------------------------------#
    threshold = 0.938

    # Learning Rate Scheduler -----------------------------------------------------#
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[30, 50],
        verbose=True
    )

    # Training process ------------------------------------------------------------#
    history = train_model(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=NUM_EPOCHS,
        device=device,
        stop_after_test_threshold=True,
        threshold=threshold,
        test_dataloader=test_dataloader,
        seed=SEED,
        qat=qat,
        bitwidth=bitwidth,
    )

    # Resulting accuracy
    print("Accuracy without qat")
    test_model(
        model=model,
        dataloader=test_dataloader,
        qat=False
    )

    if qat:
        print("Accuracy with qat")
        res_accuracy = test_model(
            model=model,
            dataloader=test_dataloader,
            qat=qat,
            bitwidth=bitwidth
        )

    # Model saving ===========================================
    save_model(
        model=model,
        model_name=model_name,
        info={
            "seed": SEED,
            "model_achitecture": model_name,
            "epoch": NUM_EPOCHS,
            "accuracy": res_accuracy
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--model", default="ModelM5", type=str)

    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--bitwidth", default=2, type=int)

    parser.add_argument("--filename", default="model_from_train", type=str)

    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    validate_params(args)

    print("Training settings: ")
    print(args)

    run(
        epochs=args.epochs,
        kernel_size=args.kernel_size,
        seed=args.seed,
        qat=args.qat,
        bitwidth=args.bitwidth,
        filename=args.filename,
        model_name=args.model,
    )
