"""Data preprocessing package for Deepfake Detection.

Provides face alignment and landmark detection utilities built on Dlib.
"""

from .face_align import FaceAligner, align_face

__all__ = ["FaceAligner", "align_face"]
