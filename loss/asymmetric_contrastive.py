"""Asymmetric Contrastive Loss for Deepfake Detection.

Standard contrastive losses treat positive and negative pairs symmetrically.
In deepfake detection, real–real pairs and real–fake pairs carry very
different semantics.  The Asymmetric Contrastive Loss (ACL) addresses this
by applying separate temperature and margin hyper-parameters to each pair
type, encouraging the model to:

  * cluster real faces tightly together, and
  * push fake faces away from real faces with a larger margin.

Mathematical formulation
------------------------
Given a mini-batch of embeddings ``z`` (shape ``B × d``) with binary labels
``y`` (0 = real, 1 = fake), we compute cosine similarities and apply:

    L = λ_r · L_real + λ_f · L_fake

where

* ``L_real``  — InfoNCE-style loss over real–real positive pairs.
* ``L_fake``  — margin-based contrastive loss pushing fake embeddings
  away from real embeddings.

Usage
-----
>>> from loss.asymmetric_contrastive import AsymmetricContrastiveLoss
>>> criterion = AsymmetricContrastiveLoss(tau_real=0.07, tau_fake=0.05, margin=0.4)
>>> loss = criterion(embeddings, labels)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricContrastiveLoss(nn.Module):
    """Asymmetric Contrastive Loss for binary real/fake classification.

    Parameters
    ----------
    tau_real:
        Temperature for the real–real InfoNCE term.  Smaller values
        produce sharper distributions.
    tau_fake:
        Temperature for the real–fake contrastive term.
    margin:
        Cosine similarity margin ``m``.  Fake embeddings whose cosine
        similarity to any real embedding exceeds ``1 - margin`` are
        penalised.
    lambda_real:
        Weight of the real–real InfoNCE loss component.
    lambda_fake:
        Weight of the real–fake margin-contrastive loss component.
    eps:
        Small constant for numerical stability.
    """

    def __init__(
        self,
        tau_real: float = 0.07,
        tau_fake: float = 0.05,
        margin: float = 0.4,
        lambda_real: float = 1.0,
        lambda_fake: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if tau_real <= 0 or tau_fake <= 0:
            raise ValueError("Temperature values must be positive.")
        if not 0.0 < margin < 2.0:
            raise ValueError("margin must be in (0, 2).")

        self.tau_real = tau_real
        self.tau_fake = tau_fake
        self.margin = margin
        self.lambda_real = lambda_real
        self.lambda_fake = lambda_fake
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Asymmetric Contrastive Loss.

        Parameters
        ----------
        embeddings:
            L2-normalised feature vectors of shape ``(B, d)``.
        labels:
            Integer class labels of shape ``(B,)``.  0 = real, 1 = fake.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        real_mask = labels == 0  # (B,)
        fake_mask = labels == 1  # (B,)

        loss_real = self._real_infonce(embeddings, real_mask)
        loss_fake = self._fake_margin(embeddings, real_mask, fake_mask)

        return self.lambda_real * loss_real + self.lambda_fake * loss_fake

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _real_infonce(
        self,
        embeddings: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE loss over real–real positive pairs.

        For each real anchor we treat all other real samples as positives
        and all fake samples as negatives.
        """
        real_embs = embeddings[real_mask]  # (R, d)
        n_real = real_embs.size(0)
        if n_real < 2:
            return embeddings.new_zeros(())

        # (R, R) cosine similarity matrix.
        sim = real_embs @ real_embs.T / self.tau_real

        # Mask out self-similarity on the diagonal.
        eye = torch.eye(n_real, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))

        # For each row the positives are all off-diagonal real entries.
        # We use the log-sum-exp trick: loss = -mean(log softmax along positives).
        log_probs = F.log_softmax(sim, dim=-1)  # (R, R)
        pos_log_probs = log_probs.masked_fill(eye, 0.0)  # zero out diagonal
        n_pos = n_real - 1
        loss = -pos_log_probs.sum(dim=-1) / n_pos  # (R,)
        return loss.mean()

    def _compute_fake_margin_loss(
        self,
        sim: torch.Tensor,
    ) -> torch.Tensor:
        """Margin loss: penalise fake–real cosine similarity > (1 - margin)."""
        # sim: (F, R)  — cosine similarity between each fake and each real.
        threshold = 1.0 - self.margin
        # hinge: max(0, sim - threshold)
        loss_per_pair = F.relu(sim - threshold)
        return loss_per_pair.mean()

    def _fake_margin(
        self,
        embeddings: torch.Tensor,
        real_mask: torch.Tensor,
        fake_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive margin loss between fake and real embeddings."""
        real_embs = embeddings[real_mask]   # (R, d)
        fake_embs = embeddings[fake_mask]   # (F, d)

        if real_embs.size(0) == 0 or fake_embs.size(0) == 0:
            return embeddings.new_zeros(())

        # (F, R) cosine similarity (embeddings already L2-normalised).
        sim = (fake_embs @ real_embs.T) / self.tau_fake
        return self._compute_fake_margin_loss(sim)
