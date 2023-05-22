import os
import sys
import json
import time
from datetime import datetime
import torch
from typing import Union

from ..models import ModelM5

from ..quantization.quant_tools import quant_forward_train, quant_forward_eval

__all__ = [
    "time_it",
    "train_model",
    "test_model",
    "save_model",
    "load_model",
    "model_size",
]


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        res_time = (time.time() - start_time) / 60
        print(
            f"Time consumption of func {func.__name__} = {res_time:.2f} mins")

    return wrapper


# TODO: Implement learning rate scheduler support
@time_it
def train_model(
        model: torch.nn.Module,
        dataloader,
        loss_fn,
        optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs,
        device='cpu',
        qat: bool = False,
        bitwidth: int = 8,
        stop_after_test_threshold: bool = False,
        test_dataloader: torch.utils.data.DataLoader = None,
        threshold: float = 0.9955,
        seed: int = None) -> list:
    """
    Implements training loop for model
    Returns history of loss and accuracy during epochs
    """
    if qat:
        print(f"Bitwidth for QAT: {bitwidth}")

    history = {'loss': [], 'accuracy': []}

    steps_per_epoch = len(dataloader)
    model.train()
    model.to(device)

    n_total = len(dataloader.dataset)

    checkpoint_version = 1

    for epoch in range(epochs):
        cum_loss = 0
        n_correct = 0
        start_time = time.time()
        for (features, labels) in dataloader:

            # Move batches to device (cuda if available)
            features = features.to(device)
            labels = labels.to(device)

            # Make predictions and calculate loss
            # If qat == True, train model, using Quantization aware training
            # if qat:
            #     predictions = quant_forward_train(
            #         model, features, bitwidth, device=device)
            # else:
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            cum_loss += loss.item()

            # Check accuracy on train dataset
            # FIXME: Add softmax
            class_predictions = torch.argmax(input=predictions, dim=1)
            n_correct += (labels == class_predictions).sum().item()

            # Calculate gradient and make gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step(
            epoch=epoch
        )

        # Save loss and accuracy
        history['loss'].append(cum_loss / steps_per_epoch)
        history['accuracy'].append(n_correct / n_total)

        # Loss tracking between different epochs
        print(f"Epoch {epoch+1}")
        print("-------------------------------")
        print(f"Average loss = {cum_loss / steps_per_epoch:.5f}")

        # Accuracy tracking
        print(f"Average accuracy = {n_correct / n_total:.5f}")

        # Accuracy on test dataset
        if test_dataloader != None:
            print("Accuracy on test dataset")
            test_acc = test_model(
                model=model,
                dataloader=test_dataloader,
                qat=qat,
                bitwidth=bitwidth,
                device=device
            )
            print(test_acc)

            if stop_after_test_threshold:
                if test_acc >= threshold:
                    print(f"Model reached needed threshold: {threshold}")
                    save_model(
                        model=model,
                        model_name=f"BestModel_V{checkpoint_version}",
                        info={
                            "seed": seed,
                            "model_achitecture": model._get_name(),
                            "epochs": epoch+1
                        }
                    )
                    checkpoint_version += 1

        # Time tracking
        res_time = time.time() - start_time

        print(f"Time for this epoch = {res_time:.2f} secs")
        print("or in minutes")
        res_time_mins = res_time / 60
        print(f"Time for this epoch = {res_time_mins:.2f} mins")

        start_date = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        print(f"Start time: {start_date}")

        end_date = datetime.fromtimestamp(time.time()).strftime("%H:%M:%S")
        print(f"End time: {end_date}")

        print("-------------------------------")


def test_model(model: torch.nn.Module, dataloader, qat: bool = False, bitwidth: int = 8, device: str = 'cuda') -> None:
    """
    This function implements accuracy measuring on passed dataloader (cuda)
    Test function with quantized weights, if qat == True
    """
    model.to(device)
    model.eval()

    n_correct = 0
    n_total = len(dataloader.dataset)

    with torch.no_grad():
        for (images, labels) in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if qat:
                logits = quant_forward_eval(
                    model=model,
                    features_batch=images,
                    bit_width=bitwidth,
                    device=device
                )
            else:
                logits = model(images)

            # predictions = torch.log_softmax(logits, dim=1)
            predictions = torch.argmax(input=logits, dim=1)
            n_correct += (labels == predictions).sum().item()

    print(f"Accuracy = {n_correct / n_total*100:.2f}%")
    return n_correct / n_total


# TODO: Rewrite to save more info
def save_model(model: torch.nn.Module, model_name: str, info: dict):
    if os.path.exists(path="checkpoints"):
        if os.path.exists(path=f"checkpoints/{model_name}/{model_name}.pt"):
            print("Rewrite old model data")
        else:
            os.mkdir(f"checkpoints/{model_name}")
        torch.save(
            obj=model.state_dict(),
            f=f"checkpoints/{model_name}/{model_name}.pt"
        )

        with open(f"checkpoints/{model_name}/{model_name}.info.json", "w") as file:
            json.dump(
                obj=info,
                fp=file
            )
    else:
        raise FileNotFoundError("Not found folder checkpoints")


def load_model(filename: str, architecture: type) -> torch.nn.Module:
    """
    Filename - file, containing state dict from saved model (without folder name and extension)
    """
    model = architecture()
    try:
        model.load_state_dict(
            torch.load(
                "params/" + filename + ".pt"
            )
        )
    except AttributeError:
        raise TypeError(
            f"architecture should be model (check models in model.py), not {architecture.__name__}")

    return model


def model_size(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size/1e3)
    os.remove('temp.p')
    return size
