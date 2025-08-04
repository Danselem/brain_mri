from pathlib import Path

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from prefect import task, flow

from src import logger
from src.inspect import visualize_batch
from src.train import train_model, tune
from src.utils import TumorClassifier, get_dataloaders

params_file = Path("params.yaml")
config = yaml.safe_load(open(params_file, encoding="utf-8"))

data_path = config["data"]["raw_data_path"]


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)



model_path = Path(config["artifacts"]["model_path"])

# Load dataset

train_dataset, train_loader, val_dataset, val_loader = get_dataloaders(data_path=data_path, batch_size=16, num_workers=0)


visualize_batch(train_loader, train_dataset)


best_params = tune(TumorClassifier, train_loader, val_loader, device)

final_model = TumorClassifier()
final_model.load_state_dict(torch.load(model_path))
final_model.to(device)

final_params = train_model(
    final_model, train_loader, val_loader, device, best_params, register=True
)