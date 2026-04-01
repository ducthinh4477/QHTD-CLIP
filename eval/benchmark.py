"""Cross-dataset AUC benchmarking utilities for Deepfake Detection.

Usage
-----
>>> from eval.benchmark import evaluate_auc, cross_dataset_benchmark
>>> auc = evaluate_auc(y_true, y_score)
>>> results = cross_dataset_benchmark(model, dataset_loaders, device="cuda")
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute AUC and related metrics for binary deepfake detection.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = real, 1 = fake), shape ``(N,)``.
    y_score:
        Predicted probability scores for the *fake* class, shape ``(N,)``.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``"auc"``   — Area Under the ROC Curve (higher is better).
        * ``"eer"``   — Equal Error Rate (lower is better).
        * ``"fpr_at_tpr95"`` — False Positive Rate at 95 % True Positive Rate.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays.")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least two distinct classes.")

    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Equal Error Rate: point where FPR ≈ FNR (= 1 - TPR).
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    # FPR at TPR = 0.95.
    tpr95_idx = np.searchsorted(tpr, 0.95, side="left")
    tpr95_idx = min(tpr95_idx, len(fpr) - 1)
    fpr_at_tpr95 = float(fpr[tpr95_idx])

    return {"auc": auc, "eer": eer, "fpr_at_tpr95": fpr_at_tpr95}


@torch.no_grad()
def cross_dataset_benchmark(
    model: nn.Module,
    dataset_loaders: Dict[str, torch.utils.data.DataLoader],
    device: Optional[str] = None,
    fake_class_index: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Evaluate *model* on multiple datasets and report per-dataset AUC.

    Parameters
    ----------
    model:
        A trained PyTorch model whose ``forward`` method returns logits of
        shape ``(B, 2)`` (real logit at index 0, fake logit at index 1).
    dataset_loaders:
        Mapping from dataset name to a ``DataLoader`` that yields
        ``(images, labels)`` batches.  Labels are expected to be binary
        (0 = real, 1 = fake).
    device:
        Target device (``"cuda"`` or ``"cpu"``).  Auto-detected when
        ``None``.
    fake_class_index:
        Index of the fake class in the model's output logits.  Default 1.

    Returns
    -------
    dict
        Nested dict ``{dataset_name: {"auc": ..., "eer": ..., "fpr_at_tpr95": ...}}``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    results: Dict[str, Dict[str, float]] = {}

    for dataset_name, loader in dataset_loaders.items():
        all_labels: list = []
        all_scores: list = []

        for images, labels in loader:
            images = images.to(device)
            logits = model(images)  # (B, 2)
            scores = torch.softmax(logits, dim=-1)[:, fake_class_index]
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())

        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores)
        results[dataset_name] = evaluate_auc(y_true, y_score)

    return results
