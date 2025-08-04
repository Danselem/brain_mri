import io
import os
import random

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import evaluate, plot_metric


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class TumorClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.0):
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
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_and_log(
    train_loader, val_loader, device, num_classes=4, epochs=10, lr=1e-3, dropout=0.1
):
    set_seed(42)

    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_experiment("TumorClassifier_NoTuning")

    class_names = list(train_loader.dataset.class_to_idx.keys())

    with mlflow.start_run():
        model = TumorClassifier(num_classes=num_classes, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_val_acc = 0.0
        best_model_state = None

        train_loss_hist, val_loss_hist = [], []
        train_acc_hist, val_acc_hist = [], []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_labels, all_preds)

            val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc = (
                evaluate(model, val_loader, criterion, device, num_classes)
            )

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": float(val_acc),
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_roc_auc": val_roc_auc,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

            print(
                f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
            )

        # Plot and log loss/accuracy history
        plot_metric(train_loss_hist, val_loss_hist, "Loss", "loss_history.png")
        plot_metric(train_acc_hist, val_acc_hist, "Accuracy", "accuracy_history.png")

        mlflow.log_artifact("loss_history.png")
        mlflow.log_artifact("accuracy_history.png")

        # Save and log best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, "best_model.pth")
            mlflow.log_artifact("best_model.pth")

            example_input = torch.randn(1, 3, 224, 224).cpu()
            try:
                mlflow.pytorch.log_model(
                    model,
                    name="model",
                )  # input_example=example_input
            except Exception as e:
                print(f"Failed to log model: {e}")

        print(f"Best validation accuracy: {best_val_acc:.4f}")
