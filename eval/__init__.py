"""Evaluation package for Deepfake Detection.

Provides cross-dataset AUC benchmarking utilities.
"""

from .benchmark import evaluate_auc, cross_dataset_benchmark

__all__ = ["evaluate_auc", "cross_dataset_benchmark"]
