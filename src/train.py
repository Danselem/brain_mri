import os
import tempfile
from pathlib import Path
from prefect import task, flow
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src import logger
from src.utils import config_mlflow, safe_set_experiment, plot_history

# Load configuration parameters
params_file = Path("params.yaml")
config = yaml.safe_load(open(params_file, encoding="utf-8"))

model_path = config["artifacts"]["model_path"]

@task(name="train_model", log_prints=True)
def train_model(model, train_loader, val_loader, device, params, register=False):
    class_names = list(train_loader.dataset.class_to_idx.keys())
    num_classes = len(class_names)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=params["lr_decay"]
    )

    best_val_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Train metrics
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(
            all_train_labels, all_train_preds, average="macro", zero_division=0
        )
        train_recall = recall_score(
            all_train_labels, all_train_preds, average="macro", zero_division=0
        )
        train_f1 = f1_score(
            all_train_labels, all_train_preds, average="macro", zero_division=0
        )

        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("train_precision", train_precision, step=epoch)
        mlflow.log_metric("train_recall", train_recall, step=epoch)
        mlflow.log_metric("train_f1", train_f1, step=epoch)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Val metrics
        val_precision = precision_score(
            all_val_labels, all_val_preds, average="macro", zero_division=0
        )
        val_recall = recall_score(
            all_val_labels, all_val_preds, average="macro", zero_division=0
        )
        val_f1 = f1_score(
            all_val_labels, all_val_preds, average="macro", zero_division=0
        )

        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
    plot_history(train_losses, val_losses, val_accuracies)
    mlflow.log_artifact("reports/images/history.png")
    if register:
        # mlflow.pytorch.log_model(model, name="model", registered_model_name="TumorClassifier")
        with tempfile.TemporaryDirectory() as tmp_dir:
            import shutil

            model_save_path = f"{tmp_dir}/model"
            mlflow.pytorch.save_model(model, path=model_save_path)

            # Step 2: Log it manually as an artifact (NOT using log_model)
            mlflow.log_artifacts(model_save_path, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="TumorClassifier")
        logger.info("Registered best model as 'TumorClassifier'.")

    return best_val_acc

@flow(name="tune_model", log_prints=True)
def tune(model_fn, train_loader, val_loader, device):
    search_space = [
        {"lr": 1e-3, "lr_decay": 0.9, "epochs": 10},
        {"lr": 1e-2, "lr_decay": 0.9, "epochs": 10},
        {"lr": 1e-2, "lr_decay": 0.95, "epochs": 10},
    ]

    best_acc = 0.0
    best_params = None

    config_mlflow()
    # mlflow.set_experiment("TumorClassifier")
    safe_set_experiment("TumorClassifier")


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
