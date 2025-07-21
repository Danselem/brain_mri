# 🧠 Brain Tumor Classification with MRI Scans

This repository contains a machine learning project for classifying brain tumors using MRI images. The model is trained to detect and categorize four types of brain conditions from axial brain scan images.

## 🗂️ Dataset

The dataset is sourced from [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and contains T1-weighted contrast-enhanced MRI images of the brain, categorized into the following classes:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

Each class has its own subdirectory containing JPEG images.

## 📁 Project Structure

```
brain-tumor-classification/
│
├── data/                  # Contains the raw and processed image data
│
├── notebooks/             # Jupyter notebooks for EDA and model experiments
│
├── src/                   # Source code for preprocessing, training, and evaluation
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── model.py
│
├── outputs/               # Saved models and evaluation metrics
│
├── requirements.txt       # Python dependencies
│
└── README.md              # Project overview and setup instructions
```

## 🧪 Model Training

This project uses deep learning (e.g., CNNs like VGG16, ResNet) for image classification. Images are preprocessed (resized, normalized), and the dataset is split into training, validation, and test sets.

You can start training using:

```bash
python src/train.py
```

Model performance is evaluated on the test set using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

## 📊 Evaluation

- Accuracy and loss plots
- Confusion matrix
- ROC curves (if applicable)

Performance metrics are saved in the `outputs/` directory after each run.

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` folder.

## 📌 Goals

- Classify MRI brain scans into one of four categories.
- Compare different CNN architectures.
- Improve accuracy through data augmentation and transfer learning.
- Save the trained model for deployment or inference.

## 🧠 Example Predictions

| MRI Image | Predicted Class |
|-----------|-----------------|
| ![glioma](assets/glioma_example.jpg) | Glioma |
| ![no tumor](assets/notumor_example.jpg) | No Tumor |

## 📚 References

- Masoud Nickparvar, Brain Tumor MRI Dataset – [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Related works on medical image classification with deep learning

## 📄 License

This project is for educational and research purposes only. Please refer to the dataset's license on Kaggle for usage terms.

---

Made with 💻 by [Your Name]

