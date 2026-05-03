# 🔬 Cervical Cell Classification (Pankaj Palwar Project)

---

## 📜 Project Proposal: CNN-Based Automated Triage of Cervical Cells

**Pankaj Palwar**  
ID: **20221186**

### Problem Statement
Manual examination of Pap smear slides is tedious, highly subjective, and prone to human error due to visual fatigue. Some cells exist in transitional states that make classification clinically challenging, especially distinguishing Benign from Abnormal cells.  
This project aims to **automate cervical cell classification** using a **custom CNN model** trained on **pre-cropped microscopic cell images**. This is a **first step** towards building a reliable secondary screening tool to assist cytopathologists in detecting precancerous lesions early.

---

### Input/Output:

- **Input**: Pre-cropped RGB microscopic image of a single cervical cell (resized to 128×128 pixels, `.bmp` format from the SIPaKMeD database).
- **Output**: Predicted class label — one of **Abnormal**, **Benign**, or **Normal**.

---

### 📊 Data Source

- **Dataset**: [SIPaKMeD Database (IEEE ICIP 2018)](https://www.cs.uoi.gr/~marina/sipakmed.html) — 4,049 pre-cropped, fully annotated cervical cell images across 5 original classes, grouped into 3 clinical categories.

---

### 🏛️ Model Architecture

Our CNN consists of **four convolutional blocks** (Conv2D → BatchNorm → ReLU → MaxPool), followed by **three fully connected layers** with Batch Normalisation and Dropout regularisation.

- **Channels Progression**: 3 → 32 → 64 → 128 → 256
- **Dropout**: 50% after FC1, 30% after FC2
- **Output**: 3 classes predicted using raw logits (CrossEntropyLoss handles Softmax internally)

This architecture uses BatchNorm after every layer for training stability, and data augmentation (random flips, rotations) to improve generalisation on the small medical dataset.

---

### 🎯 Justification

CNNs are ideal for microscopic cell images — they automatically extract spatial hierarchies of morphological features such as nuclear shape, cytoplasm texture, and cell boundary irregularities, without requiring manual feature engineering.  
This model directly addresses the image processing challenge of fine-grained visual classification in a clinical context, where the cost of misclassification (especially false negatives for Abnormal cells) is high.

---

## 📂 Project Structure

```
/project_pankaj_palwar/
│
├── checkpoints/                  # Saved model weights
│     └── final_weights.pth       # Best model (saved at peak validation accuracy)
│
├── data/                         # 30 example images (10 per class, raw .bmp format)
│     ├── im_Dyskeratotic_****.bmp     # Abnormal class (10 images)
│     ├── im_Metaplastic_****.bmp      # Benign class (10 images)
│     └── im_Parabasal_****.bmp        # Normal class (10 images)
│
├── config.py                     # All hyperparameters and path settings
├── dataset.py                    # Custom Dataset class and dataloader utility
├── model.py                      # CNN model definition (CustomCervicalCNN)
├── train.py                      # Training loop and evaluation function
├── predict.py                    # Inference script (classify_cells + inferloader)
├── main.py                       # Master run script — trains, evaluates, and plots
├── interface.py                  # Standardised interface for grading program
└── README.md                     # 📄 (this file)
```

---

## 🎯 Target Classes

The model is trained to classify cervical cells into **3 clinical categories** (mapped from 5 original SIPaKMeD classes):

| Class | SIPaKMeD Original Classes | Clinical Meaning |
|:------|:--------------------------|:-----------------|
| **Abnormal** | Dyskeratotic, Koilocytotic | Pathological — possible HPV/precancer |
| **Benign** | Metaplastic | Natural transformation zone cells |
| **Normal** | Parabasal, Superficial-Intermediate | Healthy cells |

---

## ⚙️ Configuration (`config.py`)

| Setting | Value |
|:--------|:------|
| Dataset Path | `SIPaKMeD_3_Classes/` (or set `DATASET_PATH` env variable) |
| Image Size | `(128, 128)` |
| Input Channels | `3` (RGB) |
| Batch Size | `32` |
| Shuffle Data | `True` |
| Epochs | `70` |
| Learning Rate | `0.001` |
| Weight Decay | `1e-4` (L2 regularisation) |
| Train/Val Split | `80% / 20%` |
| Checkpoint Path | `checkpoints/final_weights.pth` |
| CNN Channels | 32 → 64 → 128 → 256 |
| FC Layers | 512 → 128 → 3 |

---

## 🧠 Model Architecture (`model.py`)

A custom **Convolutional Neural Network** (`CustomCervicalCNN`) built from scratch:

### CNN Blocks:
- **Block 1**: Conv2D (3 → 32) → BatchNorm → ReLU → MaxPool (2×2)
- **Block 2**: Conv2D (32 → 64) → BatchNorm → ReLU → MaxPool (2×2)
- **Block 3**: Conv2D (64 → 128) → BatchNorm → ReLU → MaxPool (2×2)
- **Block 4**: Conv2D (128 → 256) → BatchNorm → ReLU → MaxPool (2×2)

### Fully Connected Layers:
- Flatten after final conv block
- FC1: Linear → 512 → BatchNorm → ReLU → Dropout (50%)
- FC2: Linear → 128 → BatchNorm → ReLU → Dropout (30%)
- Output Layer: Linear → `3` (classes)

---

## 🚀 How to Run the Project

### Step 1 — Setup

Place the `SIPaKMeD_3_Classes/` dataset folder inside the project directory:

```
project_pankaj_palwar/
└── SIPaKMeD_3_Classes/
    ├── Abnormal/
    ├── Benign/
    └── Normal/
```

Install dependencies:

```bash
pip install torch torchvision pillow tqdm scikit-learn matplotlib seaborn
```

---

### Step 2 — Train the Model (`main.py`)

`main.py` is the master run script that handles the complete training pipeline end-to-end:
- Loads and stratified-splits the dataset (80% train / 20% validation)
- Builds the model, loss function, and Adam optimizer
- Runs the full training loop for 70 epochs via `train.py`
- Saves the best model weights to `checkpoints/final_weights.pth` whenever validation accuracy improves
- Evaluates on the validation set after training completes
- Saves a `confusion_matrix.png` to the project directory

```bash
python main.py
```

---

### Step 3 — Run Inference (`predict.py`)

`predict.py` loads the trained weights and classifies a batch of `.bmp` images:
- `inferloader()` converts a list of image paths into a batched tensor
- `classify_cells()` runs the full batch through the model and returns predicted class label strings
- Works directly on images stored in the `data/` directory

```bash
python predict.py
```

To classify your own images in Python:

```python
from predict import classify_cells

results = classify_cells([
    "data/im_Dyskeratotic_001_01.bmp",
    "data/im_Metaplastic_001_01.bmp",
    "data/im_Parabasal_001_01.bmp"
])
print(results)  # ['Abnormal', 'Benign', 'Normal']
```

---

## 🔗 Interface (`interface.py`)

The grading program uses `interface.py` exclusively. All 7 required names are aliased correctly:

```python
from model   import CustomCervicalCNN      as TheModel
from train   import train_model            as the_trainer
from predict import classify_cells         as the_predictor
from dataset import CervicalCellDataset    as TheDataset
from dataset import get_dataloader         as the_dataloader
from config  import batch_size             as the_batch_size
from config  import epochs                 as total_epochs
```

---

## 🔍 Prediction Pipeline (`predict.py`)

**`inferloader(list_of_image_paths)`** — converts a list of `.bmp` image paths into a single batched tensor:

```python
def inferloader(list_of_image_paths):
    images = []
    for image_path in list_of_image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = _infer_transform(image)   # resize + normalise
        images.append(image_tensor)
    return torch.stack(images, dim=0)            # shape: (N, 3, 128, 128)
```

**`classify_cells(list_of_image_paths)`** — batches all images and returns predicted class label strings:

```python
def classify_cells(list_of_image_paths, model_path=config.checkpoint_path):
    image_batch = inferloader(list_of_image_paths).to(device)
    with torch.no_grad():
        logits = model(image_batch)
        _, predicted_indices = torch.max(logits, 1)
    labels = [_classes[idx] for idx in predicted_indices.cpu().numpy()]
    return labels
```

---

## 🧪 Sample Run of `predict.py`

```
Model weights loaded from checkpoints/final_weights.pth
  data/im_Dyskeratotic_052_01.bmp  →  Abnormal
  data/im_Metaplastic_059_01.bmp   →  Benign
  data/im_Parabasal_031_03.bmp     →  Normal

Results: ['Abnormal', 'Benign', 'Normal']
```

---

## 📈 Training Results (70 Epochs)

| Metric | Value |
|:-------|:------|
| Best Validation Accuracy | **97.41%** (Epoch 69) |
| Final Validation Accuracy | 96.67% |
| Final Validation Loss | 0.0879 |
| Best Validation Loss | 0.0810 |

### Per-Class Accuracy (Confusion Matrix):

| Class | Correct | Total | Accuracy |
|:------|:--------|:------|:---------|
| Abnormal | 315 | 328 | 96.04% |
| Benign | 147 | 159 | 92.45% |
| Normal | 322 | 324 | 99.38% |
| **Overall** | **784** | **811** | **96.67%** |

---
