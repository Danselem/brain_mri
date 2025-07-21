import shutil
import kagglehub
from pathlib import Path

# Download dataset
cached_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

# Define your desired destination
destination = Path("data/raw/brain-tumor-mri")

destination.mkdir(parents=True, exist_ok=True)

# Move dataset contents to your custom path
shutil.copytree(cached_path, destination, dirs_exist_ok=True)

print("Dataset copied to:", destination.resolve())
