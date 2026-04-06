# QHTD-CLIP: effort_asy for Deepfake Detection

Du an cuoi ky mon Toan cho Tri tue Nhan tao (HK2 2025-2026), tap trung vao mot model duy nhat: `effort_asy`.

Muc tieu cua repo nay:
- Giu pipeline train/test cho effort_asy tu DeepfakeBench.
- Loai bo cac detector/chuc nang benchmark khong lien quan.
- Duy tri kha nang tai lap ket qua voi du lieu da preprocess san.

## Thong tin nhom
- Truong: Dai hoc Su pham Ky thuat TP.HCM (HCMUTE)
- Sinh vien: Nguyen Duc Thinh, Phung Le Thanh Quan

## Tom tat phuong phap
- Backbone: CLIP ViT-L/14 (`openai/clip-vit-large-patch14`).
- Ky thuat tinh chinh: SVD residual adaptation tren cac lop self-attention.
- Head: Cosine classifier.
- Loss: Cross Entropy + Asymmetric Contrastive Loss.

Cong thuc tong quat:

$$
\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{asym}
$$

## Cau truc repo
- `DeepfakeBench/`: source code chay train/test effort_asy.
- `ckpt_best.pth`: checkpoint da huan luyen.
- `Nhom01_TLCK.pdf`: bao cao cuoi ky.

## Yeu cau du lieu
Repo nay gia dinh du lieu da preprocess san:
- Thu muc anh: `DeepfakeBench/datasets/rgb` (co the sua trong config).
- JSON metadata: `DeepfakeBench/preprocessing/dataset_json`.

File cau hinh lien quan:
- `DeepfakeBench/training/config/detector/effort_asy.yaml`
- `DeepfakeBench/training/config/train_config.yaml`
- `DeepfakeBench/training/config/test_config.yaml`

## Cai dat moi truong
Xem huong dan nhanh trong `conda.txt`.

Neu da o thu muc root repo:

```bash
cd DeepfakeBench
bash install.sh
```

## Lenh train
Tu thu muc root repo:

```bash
cd DeepfakeBench
python training/train.py \
  --detector_path ./training/config/detector/effort_asy.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0"
```

## Lenh test
Tu thu muc root repo:

```bash
cd DeepfakeBench
python training/test.py \
  --detector_path ./training/config/detector/effort_asy.yaml \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0" \
  --weights_path ../ckpt_best.pth
```

## Ket qua tham khao
- Cross-dataset AUC trung binh: 0.9392
- In-domain AUC trung binh (FF++): 99.05
- So tham so train duoc: 0.19M

## Luu y tai lap
- Gan duong dan du lieu dung trong `train_config.yaml` va `test_config.yaml`.
- Lan dau chay se tai CLIP tu HuggingFace.
- Neu khong co Internet, can cache model truoc.

## Citation
```bibtex
@article{nguyenphung2026qhtd,
  title={Ung dung SVD bao toan khong gian dac trung khi tai huan luyen mo hinh cho bai toan phan loai anh gia khuon mat},
  author={Nguyen Duc Thinh and Phung Le Thanh Quan},
  journal={Tieu luan ket thuc hoc phan Toan cho Tri tue Nhan tao - HCMUTE},
  year={2026}
}
```