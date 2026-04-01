"""Models package for Deepfake Detection.

Contains CLIP-based backbone with SVD-PEFT (Parameter-Efficient Fine-Tuning
via Singular Value Decomposition) adapters.
"""

from .clip_svd_peft import CLIPWithSVDPEFT, SVDAdapter

__all__ = ["CLIPWithSVDPEFT", "SVDAdapter"]
