"""CLIP backbone with SVD-PEFT (Singular Value Decomposition Parameter-Efficient
Fine-Tuning) adapters for Deepfake Detection.

The SVD-PEFT approach decomposes each trainable weight matrix ``W`` as

    W_adapted = W_0 + ΔW,   ΔW = U · diag(s) · V^T

where ``W_0`` is frozen, and only the low-rank singular triplet
``(U, s, V)`` is trained.  This yields far fewer trainable parameters
compared to full fine-tuning while preserving the CLIP pre-training.

Usage
-----
>>> from models.clip_svd_peft import CLIPWithSVDPEFT
>>> model = CLIPWithSVDPEFT(clip_model_name="ViT-B/32", rank=8)
>>> logits = model(images)          # (B, 2) real / fake logits
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SVD-PEFT adapter layer
# ---------------------------------------------------------------------------


class SVDAdapter(nn.Module):
    """A low-rank residual adapter using truncated SVD parametrisation.

    For an input linear layer of shape ``(out_features, in_features)``, the
    adapter adds a residual ``ΔW = U · diag(s) · V^T`` with rank ``r``.

    Parameters
    ----------
    in_features:
        Input dimension of the frozen linear layer.
    out_features:
        Output dimension of the frozen linear layer.
    rank:
        Rank ``r`` of the SVD decomposition.  Must satisfy
        ``r <= min(in_features, out_features)``.
    alpha:
        Scaling factor applied to ``ΔW`` before adding it.  Defaults to
        ``rank`` (i.e. the effective learning rate stays independent of
        ``rank``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: Optional[float] = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) must be <= min(in_features, out_features) "
                f"({min(in_features, out_features)})"
            )

        self.rank = rank
        self.scaling = (alpha if alpha is not None else float(rank)) / rank

        # U: (out_features, rank)  — initialised with Kaiming uniform
        # s: (rank,)               — initialised to ones
        # V: (in_features, rank)   — initialised with Kaiming uniform
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.s = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))

        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        """Compute the residual weight matrix ``ΔW``."""
        # (out_features, rank) @ diag(s) @ (rank, in_features)
        return self.scaling * (self.U * self.s.unsqueeze(0)) @ self.V.T

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """Apply the adapted linear transformation.

        Parameters
        ----------
        x:
            Input tensor of shape ``(..., in_features)``.
        base_weight:
            The frozen weight matrix ``W_0`` of shape
            ``(out_features, in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., out_features)``.
        """
        adapted_weight = base_weight + self.delta_weight()
        return x @ adapted_weight.T


# ---------------------------------------------------------------------------
# CLIP + SVD-PEFT model
# ---------------------------------------------------------------------------


class CLIPWithSVDPEFT(nn.Module):
    """CLIP visual encoder augmented with SVD-PEFT adapters for binary
    deepfake detection.

    The CLIP backbone is kept fully frozen; only the SVD-PEFT adapter
    parameters and the classification head are trained.

    Parameters
    ----------
    clip_model_name:
        Any model name accepted by ``clip.load`` (e.g. ``"ViT-B/32"``).
    rank:
        SVD rank used for all adapter layers.
    alpha:
        Scaling factor for adapter outputs. Defaults to ``rank``.
    device:
        Target device string (``"cuda"`` or ``"cpu"``).  Auto-detected when
        ``None``.
    adapter_targets:
        Names of sub-modules (Linear layers) inside the CLIP visual encoder
        to attach adapters to.  When ``None``, adapters are attached to all
        ``nn.Linear`` layers in the visual encoder.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        rank: int = 8,
        alpha: Optional[float] = None,
        device: Optional[str] = None,
        adapter_targets: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        try:
            import clip  # openai/CLIP
        except ImportError as exc:
            raise ImportError(
                "The 'clip' package is required. Install it with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.visual = clip_model.visual
        self.embed_dim: int = self.visual.output_dim

        # Freeze the entire CLIP visual encoder.
        for param in self.visual.parameters():
            param.requires_grad_(False)

        # Attach SVD-PEFT adapters.
        self.adapters: nn.ModuleDict = nn.ModuleDict()
        self._adapter_targets: dict = {}  # module name -> (in_features, out_features)

        for name, module in self.visual.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if adapter_targets is not None and name not in adapter_targets:
                continue
            in_f, out_f = module.in_features, module.out_features
            r = min(rank, min(in_f, out_f))
            safe_name = name.replace(".", "_")
            self.adapters[safe_name] = SVDAdapter(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                alpha=alpha,
            )
            self._adapter_targets[name] = safe_name

        # Binary classification head: real (0) vs fake (1).
        self.classifier = nn.Linear(self.embed_dim, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_with_adapters(self, x: torch.Tensor) -> torch.Tensor:
        """Run the visual encoder, patching adapted layers on the fly.

        For each Linear layer that has an associated adapter we temporarily
        replace its ``forward`` with one that uses the adapted weight.
        """
        hooks = []
        for name, module in self.visual.named_modules():
            if name not in self._adapter_targets:
                continue
            safe_name = self._adapter_targets[name]
            adapter = self.adapters[safe_name]
            frozen_weight = module.weight.detach()

            def make_hook(adp: SVDAdapter, w: torch.Tensor, bias: Optional[torch.Tensor]):
                def hook(mod, inp, out):
                    # Re-compute with adapted weight.
                    x_in = inp[0]
                    result = adp.forward(x_in, w)
                    if bias is not None:
                        result = result + bias
                    return result

                return hook

            h = module.register_forward_hook(
                make_hook(adapter, frozen_weight, module.bias)
            )
            hooks.append(h)

        features = self.visual(x)

        for h in hooks:
            h.remove()

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Classify *images* as real or fake.

        Parameters
        ----------
        images:
            Tensor of shape ``(B, C, H, W)`` preprocessed with
            ``self.preprocess``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, 2)``.
        """
        features = self._forward_with_adapters(images)
        logits = self.classifier(features)
        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only the trainable (adapter + classifier) parameters."""
        for p in self.adapters.parameters():
            yield p
        yield from self.classifier.parameters()

    def count_parameters(self) -> dict:
        """Return a dict with total, frozen, and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "frozen": total - trainable,
            "trainable": trainable,
        }
