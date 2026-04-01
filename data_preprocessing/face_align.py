"""Face alignment using Dlib's 68-point facial landmark predictor.

Usage
-----
>>> from data_preprocessing.face_align import FaceAligner, align_face
>>> aligner = FaceAligner(predictor_path="shape_predictor_68_face_landmarks.dat")
>>> aligned = aligner.align(image)          # numpy BGR image  -> aligned BGR crop
>>> aligned = align_face(image, aligner)    # convenience wrapper
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import dlib
import numpy as np

# Reference (mean) facial landmark positions used for affine alignment.
# Coordinates are normalised to [0, 1] relative to the output crop size.
_REFERENCE_LANDMARKS_68 = np.array(
    [
        [0.19, 0.37],  # left eye centre
        [0.81, 0.37],  # right eye centre
        [0.50, 0.60],  # nose tip
        [0.30, 0.82],  # left mouth corner
        [0.70, 0.82],  # right mouth corner
    ],
    dtype=np.float32,
)

# Indices within the 68-point set corresponding to the five reference points.
_LM68_INDICES = {
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "nose_tip": [30],
    "left_mouth": [48],
    "right_mouth": [54],
}


def _mean_landmark(landmarks: dlib.full_object_detection, indices: list) -> np.ndarray:
    """Return the mean (x, y) position of a group of landmark points."""
    pts = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in indices],
        dtype=np.float32,
    )
    return pts.mean(axis=0)


class FaceAligner:
    """Aligns detected faces to a canonical pose using Dlib landmarks.

    Parameters
    ----------
    predictor_path:
        Path to Dlib's ``shape_predictor_68_face_landmarks.dat`` file.
    output_size:
        Side length (pixels) of the square output crop.  Default ``224``.
    margin:
        Fractional margin added around the tight face bounding box before
        alignment.  Default ``0.4``.
    """

    def __init__(
        self,
        predictor_path: str = "shape_predictor_68_face_landmarks.dat",
        output_size: int = 224,
        margin: float = 0.4,
    ) -> None:
        if not os.path.isfile(predictor_path):
            raise FileNotFoundError(
                f"Dlib shape predictor not found at '{predictor_path}'. "
                "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.output_size = output_size
        self.margin = margin

        # Scale reference landmarks to output pixel space.
        self._ref = _REFERENCE_LANDMARKS_68 * output_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        image: np.ndarray,
        rect: Optional[dlib.rectangle] = None,
    ) -> Optional[np.ndarray]:
        """Align a face in *image* to a canonical crop.

        Parameters
        ----------
        image:
            BGR (or grayscale) NumPy array as returned by ``cv2.imread``.
        rect:
            Optional pre-detected face bounding box (``dlib.rectangle``).
            When ``None``, the detector runs automatically and the largest
            detected face is used.

        Returns
        -------
        numpy.ndarray or None
            Aligned BGR face crop of shape ``(output_size, output_size, 3)``,
            or ``None`` if no face was detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if rect is None:
            rects = self.detector(gray, 1)
            if len(rects) == 0:
                return None
            # Pick the largest detection.
            rect = max(rects, key=lambda r: r.width() * r.height())

        landmarks = self.predictor(gray, rect)

        # Compute the five reference-point positions in the source image.
        src_pts = np.stack(
            [
                _mean_landmark(landmarks, _LM68_INDICES["left_eye"]),
                _mean_landmark(landmarks, _LM68_INDICES["right_eye"]),
                _mean_landmark(landmarks, _LM68_INDICES["nose_tip"]),
                _mean_landmark(landmarks, _LM68_INDICES["left_mouth"]),
                _mean_landmark(landmarks, _LM68_INDICES["right_mouth"]),
            ]
        )  # shape (5, 2)

        transform = cv2.estimateAffinePartial2D(src_pts, self._ref)[0]
        if transform is None:
            return None

        aligned = cv2.warpAffine(
            image,
            transform,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    def detect_and_align_all(
        self,
        image: np.ndarray,
    ) -> list:
        """Detect every face in *image* and return a list of aligned crops."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        rects = self.detector(gray, 1)
        crops = []
        for rect in rects:
            crop = self.align(image, rect=rect)
            if crop is not None:
                crops.append(crop)
        return crops


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def align_face(
    image: np.ndarray,
    aligner: FaceAligner,
    rect: Optional[dlib.rectangle] = None,
) -> Optional[np.ndarray]:
    """Thin wrapper around :meth:`FaceAligner.align`.

    Parameters
    ----------
    image:
        BGR NumPy array.
    aligner:
        A pre-constructed :class:`FaceAligner` instance.
    rect:
        Optional pre-detected bounding box.

    Returns
    -------
    numpy.ndarray or None
    """
    return aligner.align(image, rect=rect)
