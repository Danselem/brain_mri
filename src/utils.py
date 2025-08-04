import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from prefect import task, flow

from src import logger

# Load configuration parameters
# === Configuration ===
load_dotenv(Path("./.env"))
params_file = Path("params.yaml")
config = yaml.safe_load(open(params_file, encoding="utf-8"))

history_path = config["reports"]["history_path"]

Path(history_path).parent.mkdir(parents=True, exist_ok=True)

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")


def config_mlflow() -> None:
    if (
        DAGSHUB_REPO_OWNER is None
        or DAGSHUB_REPO is None
        or os.getenv("DAGSHUB_TOKEN") is None
    ):
        raise ValueError(
            "DAGSHUB_REPO_OWNER, DAGSHUB_REPO, and DAGSHUB_TOKEN environment variables must be set."
        )
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token is None:
        raise ValueError("DAGSHUB_TOKEN environment variable must be set.")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO}.mlflow"
    )

def safe_set_experiment(name):
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(name)

def promote_and_save_model(model_name: str, onnx_save_path: str):
    config_mlflow()
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for registered model '{model_name}'")

    latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
    version_number = latest_version.version

    client.set_registered_model_alias(
        name=model_name, alias="prod", version=version_number
    )

    model = mlflow.pytorch.load_model(f"models:/{model_name}@prod")
    model.eval()

    device = next(model.parameters()).device  # Get model's device
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)

    torch.onnx.export(model, dummy_input, onnx_save_path)


def plot_history(train_losses, val_losses, val_accuracies, output_path=history_path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss History")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    all_probs = torch.cat(all_probs).numpy()
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        roc_auc = 0.0

    return avg_loss, accuracy, precision, recall, f1, roc_auc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class TumorClassifier(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@task(name="load", retries=3, retry_delay_seconds=10, log_prints=True)
def get_dataloaders(data_path, batch_size=16, num_workers=0):
    """
    Creates training and validation DataLoaders from a given data path.

    Args:
        data_path (str or Path): Path to the root dataset folder.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation sets.
    """
    raw_data_path = Path(data_path)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(raw_data_path / "Training", transform=data_transforms)
    val_dataset = ImageFolder(raw_data_path / "Testing", transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, train_loader, val_dataset, val_loader
