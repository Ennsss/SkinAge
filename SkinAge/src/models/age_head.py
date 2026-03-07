"""
Age regression head for the SkinAge multi-task model.

Predicts chronological skin age (in years) from a 1408-dimensional global
feature vector produced by the EfficientNet-B2 backbone.

Architecture
------------
    Linear(1408 → 256) → ReLU → Dropout(0.3) → Linear(256 → 1) → ReLU

Output range
------------
The module outputs a non-negative scalar per sample.  ReLU at the output
layer enforces the physical constraint that age cannot be negative.

Mixed-label training
--------------------
Not every sample in a batch carries an age label (only UTKFace images do).
The training loop uses ``batch["age_indices"]`` to select the subset of
predictions that have a corresponding ground-truth label before computing
the age regression loss — see :func:`~src.data.dataset.skinage_collate_fn`
for details.  This module is unaware of that masking; it always produces a
prediction for every element in the batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AgeHead(nn.Module):
    """Regression head that maps pooled backbone features to a predicted age.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.  Must match the backbone's
        output channel depth (EfficientNet-B2 → 1408).
    hidden_dim : int
        Width of the intermediate fully-connected layer.
    dropout : float
        Dropout probability applied after the hidden layer activation.

    Inputs
    ------
    x : torch.Tensor, shape ``(B, in_features)``
        Globally pooled backbone feature vectors.

    Returns
    -------
    torch.Tensor, shape ``(B, 1)``
        Predicted age in years, guaranteed non-negative (>= 0).
    """

    def __init__(
        self,
        in_features: int = 1408,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.in_features: int = in_features
        self.hidden_dim: int = hidden_dim
        self.dropout_p: float = dropout

        self.net = nn.Sequential(
            # Hidden layer: distil global features into age-relevant factors
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Output layer: single age logit
            nn.Linear(hidden_dim, 1),
            # ReLU ensures the predicted age is always non-negative
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict chronological age for a batch of feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, in_features)``.  Globally pooled backbone output.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``.  Predicted age in years, values >= 0.
        """
        return self.net(x)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AgeHead("
            f"in_features={self.in_features}, "
            f"hidden_dim={self.hidden_dim}, "
            f"dropout={self.dropout_p})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    head = AgeHead()
    x = torch.randn(4, 1408)
    out = head(x)
    print(f"Age output: {out.shape}")  # (4, 1)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"
    assert (out >= 0).all(), (
        f"Negative age predictions found: min={out.min():.4f}"
    )
    print("Age head: all checks passed!")
