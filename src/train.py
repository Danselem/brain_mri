import os
import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import numpy as np 
from src.utils import TumorClassifier, evaluate, set_seed, plot_history
from pathlib import Path
from src import logger
import yaml

params_file = Path("params.yaml")
config = yaml.safe_load(open(params_file, encoding="utf-8"))

model_path = config["artifacts"]["model_path"]

def train_model(model, train_loader, val_loader, device, params, register=False):
    class_names = list(train_loader.dataset.class_to_idx.keys())
    num_classes = len(class_names)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=params["lr_decay"])

    best_val_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)

    if register:
        mlflow.pytorch.log_model(model, name="model", registered_model_name="TumorClassifier")
        logger.info("Registered best model as 'TumorClassifier'.")

    return best_val_acc


def tune(model_fn, train_loader, val_loader, device):
    search_space = [
        {"lr": 1e-3, "lr_decay": 0.9, "epochs": 2},
    ]

    best_acc = 0.0
    best_params = None

    mlflow.set_experiment("TumorClassifier")

    for i, params in enumerate(search_space):
        logger.info(f"\nTrial {i+1} with params: {params}")

        with mlflow.start_run(nested=True) as run:
            mlflow.log_params(params)

            model = model_fn()
            model.to(device)

            val_acc = train_model(model, train_loader, val_loader, device, params)

            logger.info(f"Validation accuracy for trial {i+1}: {val_acc}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params

    if best_params is None:
        best_params = search_space[0]

    logger.info(f"\nBest parameters found: {best_params}")
    return best_params






