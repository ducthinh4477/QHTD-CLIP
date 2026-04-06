# DeepfakeBench Subset (effort_asy only)

This folder has been trimmed to support only the effort_asy model for the QHTD-CLIP final project.

## Scope

- Kept: files needed to train and evaluate effort_asy.
- Removed: unrelated detectors, benchmark analysis scripts, optional preprocessing tools, and unused configs.

## Quick Start

From repository root:

```bash
cd DeepfakeBench
bash install.sh
```

Train:

```bash
python training/train.py \
  --detector_path ./training/config/detector/effort_asy.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0"
```

Test with provided checkpoint:

```bash
python training/test.py \
  --detector_path ./training/config/detector/effort_asy.yaml \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0" \
  --weights_path ../ckpt_best.pth
```

## Data Assumptions

This repo expects preprocessed data and metadata:

- RGB frames directory: ./datasets/rgb
- Dataset JSON directory: ./preprocessing/dataset_json

You can edit data paths in:

- training/config/train_config.yaml
- training/config/test_config.yaml

## Upstream Credit

This project is built on top of DeepfakeBench (NeurIPS 2023 Datasets and Benchmarks), but only keeps the minimum code path for effort_asy.
