import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.utils import TumorClassifier
from src.train import train_model, tune
from src.inspect import visualize_batch
from src import logger
from pathlib import Path
import yaml

params_file = Path("params.yaml")
config = yaml.safe_load(open(params_file, encoding="utf-8"))

data_path = config["data"]["raw_data_path"]

# Data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


raw_data_path = Path(data_path)

train_dataset = ImageFolder(raw_data_path / 'Training', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

val_dataset = ImageFolder(raw_data_path / 'Testing', transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

visualize_batch(train_loader, device)

best_params = tune(TumorClassifier, train_loader, val_loader, device)

final_model = TumorClassifier()
final_model.load_state_dict(torch.load("best_model.pth"))
final_model.to(device)

final_params = train_model(final_model, train_loader, val_loader, device, best_params, register=True)