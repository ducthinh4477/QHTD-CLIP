"""Loss functions package for Deepfake Detection.

Provides Asymmetric Contrastive Loss that applies different margins for
real/fake pairs to improve generalisation across unseen forgery methods.
"""

from .asymmetric_contrastive import AsymmetricContrastiveLoss

__all__ = ["AsymmetricContrastiveLoss"]
