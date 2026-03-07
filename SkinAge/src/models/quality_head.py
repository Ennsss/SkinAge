"""
Zone quality regression head for the SkinAge multi-task model.

Predicts 28 per-zone / per-concern quality scores from a 1408-dimensional
global feature vector produced by the EfficientNet-B2 backbone.

Architecture
------------
    Linear(1408 → 512) → ReLU → Dropout(0.3) → Linear(512 → 28) → Sigmoid

Output range
------------
The module outputs values in ``[0.0, 1.0]``.  This matches the label space
used by :class:`~src.data.dataset.SkinAgeDataset`, which normalises source
scores from ``[0, 100]`` to ``[0, 1]`` before training.

    Training loss  : operates in ``[0.0, 1.0]``  (matches normalised labels)
    Inference / API: caller multiplies by 100 to recover ``[0, 100]`` scores

The config annotation ``activation: sigmoid_x_100`` describes *inference*
behaviour only.  The ``x100`` scaling must NOT be applied inside this module;
doing so would break the loss computation during training.

Output ordering
---------------
Zone-major flat ordering — all four concerns for zone 0 first, then zone 1,
etc. — consistent with ``QUALITY_SCORE_COLUMNS`` in
:mod:`src.data.dataset`:

    forehead_wrinkle, forehead_pigmentation, forehead_redness,
    forehead_pore_texture, under_eyes_wrinkle, …, nasolabial_pore_texture
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QualityHead(nn.Module):
    """Regression head that maps pooled backbone features to 28 quality scores.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.  Must match the backbone's
        output channel depth (EfficientNet-B2 → 1408).
    hidden_dim : int
        Width of the intermediate fully-connected layer.
    num_zones : int
        Number of facial zones assessed (default: 7).
    num_concerns : int
        Number of concern types per zone (default: 4).
    dropout : float
        Dropout probability applied after the hidden layer activation.

    Inputs
    ------
    x : torch.Tensor, shape ``(B, in_features)``
        Globally pooled backbone feature vectors.

    Returns
    -------
    torch.Tensor, shape ``(B, num_zones * num_concerns)``
        Quality scores in ``[0.0, 1.0]``, zone-major ordering.
    """

    def __init__(
        self,
        in_features: int = 1408,
        hidden_dim: int = 512,
        num_zones: int = 7,
        num_concerns: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.in_features: int = in_features
        self.hidden_dim: int = hidden_dim
        self.num_zones: int = num_zones
        self.num_concerns: int = num_concerns
        self.num_outputs: int = num_zones * num_concerns  # 28
        self.dropout_p: float = dropout

        self.net = nn.Sequential(
            # Hidden layer: compress global features to a task-specific space
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Output layer: one logit per zone-concern pair
            nn.Linear(hidden_dim, self.num_outputs),
            # Sigmoid bounds each score to [0, 1] — matches normalised labels
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quality scores for a batch of feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, in_features)``.  Globally pooled backbone output.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_zones * num_concerns)``.
            Values in ``[0.0, 1.0]``.  Multiply by 100 at inference time to
            convert to the human-readable ``[0, 100]`` scoring range.
        """
        return self.net(x)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"QualityHead("
            f"in_features={self.in_features}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_zones={self.num_zones}, "
            f"num_concerns={self.num_concerns}, "
            f"num_outputs={self.num_outputs}, "
            f"dropout={self.dropout_p})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    head = QualityHead()
    x = torch.randn(4, 1408)
    out = head(x)
    print(f"Quality output: {out.shape}")  # (4, 28)
    assert out.shape == (4, 28), f"Expected (4, 28), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), (
        f"Scores out of [0, 1] range: min={out.min():.4f}, max={out.max():.4f}"
    )
    print("Quality head: all checks passed!")
