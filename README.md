# QHTD-CLIP
QHTD based on CLIP for Generalizable Face Forgery Detection. Research project for "Math for AI" course at HCMUTE.

## Project structure

```
QHTD-CLIP/
├── data_preprocessing/          # Face detection & alignment (Dlib)
│   ├── __init__.py
│   └── face_align.py            # FaceAligner – 68-point landmark alignment
├── models/                      # Model architectures
│   ├── __init__.py
│   └── clip_svd_peft.py         # CLIP + SVD-PEFT adapters (binary detector)
├── loss/                        # Loss functions
│   ├── __init__.py
│   └── asymmetric_contrastive.py  # Asymmetric Contrastive Loss (ACL)
├── eval/                        # Evaluation utilities
│   ├── __init__.py
│   └── benchmark.py             # Cross-dataset AUC benchmarking
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
# Install OpenAI CLIP (not on PyPI):
pip install git+https://github.com/openai/CLIP.git
# Download Dlib's facial landmark model:
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Modules

### `data_preprocessing` – Face Alignment with Dlib

Uses Dlib's 68-point facial landmark detector to align faces to a canonical
pose before feeding them into CLIP.

```python
from data_preprocessing import FaceAligner, align_face
import cv2

aligner = FaceAligner("shape_predictor_68_face_landmarks.dat", output_size=224)
image   = cv2.imread("face.jpg")
aligned = aligner.align(image)   # numpy BGR array (224, 224, 3)
```

### `models` – SVD-PEFT Implementation

Attaches low-rank SVD adapters (`SVDAdapter`) to every `nn.Linear` layer
inside CLIP's visual encoder.  Only the adapter parameters and the final
binary classification head are trained.

```python
from models import CLIPWithSVDPEFT

model  = CLIPWithSVDPEFT(clip_model_name="ViT-B/32", rank=8)
logits = model(images)   # (B, 2)  –  col 0 = real, col 1 = fake
print(model.count_parameters())
# {'total': ..., 'frozen': ..., 'trainable': ...}
```

### `loss` – Asymmetric Contrastive Loss

Applies separate temperature and margin hyper-parameters to real–real and
real–fake pairs, encouraging tight clustering of genuine faces and a large
separation for forged ones.

```python
from loss import AsymmetricContrastiveLoss

criterion = AsymmetricContrastiveLoss(tau_real=0.07, tau_fake=0.05, margin=0.4)
loss      = criterion(embeddings, labels)   # embeddings: (B, d), labels: (B,)
```

### `eval` – Cross-Dataset AUC Benchmarking

Evaluates a trained model across multiple test datasets and reports AUC,
Equal Error Rate (EER), and FPR at TPR = 95 %.

```python
from eval import evaluate_auc, cross_dataset_benchmark

# Single dataset
metrics = evaluate_auc(y_true, y_score)
# {'auc': 0.97, 'eer': 0.06, 'fpr_at_tpr95': 0.12}

# Multiple datasets
results = cross_dataset_benchmark(model, {"FF++": ff_loader, "CDF": cdf_loader})
```
