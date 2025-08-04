from pathlib import Path

import pytest
import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.inspect import visualize_batch
from src.train import train_model, tune
from src.utils import TumorClassifier


# Mock visualize_batch to avoid plotting during tests
@pytest.fixture(autouse=True)
def patch_visualize(monkeypatch):
    monkeypatch.setattr("src.inspect.visualize_batch", lambda *args, **kwargs: None)


@pytest.fixture(scope="module")
def config():
    params_file = Path("params.yaml")
    with open(params_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def dataloaders(config):
    data_path = Path(config["data"]["raw_data_path"])

    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(data_path / "Training", transform=data_transforms)
    val_dataset = ImageFolder(data_path / "Testing", transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    return train_loader, val_loader


@pytest.fixture(scope="module")
def device():
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def test_dataloaders(dataloaders):
    train_loader, val_loader = dataloaders
    images, labels = next(iter(train_loader))
    assert images.shape[0] > 0
    assert images.shape[1:] == (3, 224, 224)
    assert isinstance(labels[0].item(), int)


def test_tune_returns_params(dataloaders, device):
    train_loader, val_loader = dataloaders
    params = tune(TumorClassifier, train_loader, val_loader, device)
    assert isinstance(params, dict)
    assert "lr" in params or "dropout" in params  # Example expected keys


def test_train_model(dataloaders, device):
    train_loader, val_loader = dataloaders
    model = TumorClassifier()
    params = {
        "lr": 0.001,
        "dropout": 0.5,
        "epochs": 1,
        "lr_decay": 0.95,
    }  # Minimal for test
    result = train_model(
        model, train_loader, val_loader, device, params, register=False
    )
    assert isinstance(result, dict)
    assert "accuracy" in result or "f1_score" in result
