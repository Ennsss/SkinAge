"""
Full SkinAge multi-task model.

Assembles the four modular components into a single nn.Module:

    SkinAgeBackbone  ->  (skip_features, pooled)
                          |                |
                    UNetDecoder       QualityHead / AgeHead
                          |                |
                     heatmaps (B,4,H,W)  quality (B,28) / age (B,1)

All four sub-modules are sourced from the same ``src/models/`` package so
that this file is the single entry-point callers need to import.

Typical usage
-------------
From a YAML config (recommended)::

    model = SkinAgeModel.from_config("SkinAge/config/model_config.yaml")

From a plain dict or defaults::

    model = SkinAgeModel(config={"quality_hidden": 512}, pretrained=False)

Phase-1 / Phase-2 training pattern::

    model.freeze_backbone()      # Phase 1 – heads only
    train(model, phase1_loader)
    model.unfreeze_backbone()    # Phase 2 – full fine-tune
    train(model, phase2_loader)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass  # keep runtime import-free for type stubs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy sub-module imports (resolved at class definition time, not at the top
# of the file, so that circular-import risk across the package is zero).
# ---------------------------------------------------------------------------
from .age_head import AgeHead  # noqa: E402
from .backbone import SkinAgeBackbone  # noqa: E402
from .quality_head import QualityHead  # noqa: E402
from .unet_decoder import UNetDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Config loader helper
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return the parsed dict.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ImportError
        When PyYAML is not installed (it is a declared project dependency,
        so this should never happen in practice).
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to load model_config.yaml.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# SkinAgeModel
# ---------------------------------------------------------------------------

class SkinAgeModel(nn.Module):
    """Multi-task model for skin-age estimation and quality scoring.

    Produces three outputs per forward pass:

    ``heatmaps``
        Shape ``(B, 4, 512, 512)``.  Four spatial heatmaps — one per skin
        concern (wrinkle, pigmentation, redness, pore_texture) — decoded by
        a U-Net head from the backbone skip features.

    ``quality``
        Shape ``(B, 28)``.  Per-zone / per-concern quality scores in
        ``[0, 1]`` (scaled to ``[0, 100]`` by the QualityHead sigmoid).

    ``age``
        Shape ``(B, 1)``.  Non-negative biological skin-age prediction in
        years (ReLU output).

    Parameters
    ----------
    config:
        Optional flat dict overriding any of the following keys:

        ``decoder_channels``    list[int]   default [256, 128, 64, 32]
        ``output_channels``     int         default 4
        ``quality_hidden``      int         default 512
        ``quality_dropout``     float       default 0.3
        ``age_hidden``          int         default 256
        ``age_dropout``         float       default 0.3

        When ``None``, all defaults apply.
    pretrained:
        Pass ``True`` to initialise the backbone from ImageNet weights.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        cfg: Dict[str, Any] = config or {}

        # ------------------------------------------------------------------ #
        # Backbone                                                             #
        # ------------------------------------------------------------------ #
        self.backbone = SkinAgeBackbone(pretrained=pretrained)

        # ------------------------------------------------------------------ #
        # U-Net decoder                                                        #
        # ------------------------------------------------------------------ #
        self.decoder = UNetDecoder(
            encoder_channels=self.backbone.ENCODER_CHANNELS,
            decoder_channels=cfg.get("decoder_channels", [256, 128, 64, 32]),
            output_channels=cfg.get("output_channels", 4),
        )

        # ------------------------------------------------------------------ #
        # Regression / classification heads                                    #
        # ------------------------------------------------------------------ #
        self.quality_head = QualityHead(
            in_features=self.backbone.POOLED_DIM,  # 1408
            hidden_dim=cfg.get("quality_hidden", 512),
            dropout=cfg.get("quality_dropout", 0.3),
        )

        self.age_head = AgeHead(
            in_features=self.backbone.POOLED_DIM,  # 1408
            hidden_dim=cfg.get("age_hidden", 256),
            dropout=cfg.get("age_dropout", 0.3),
        )

        logger.info(
            "SkinAgeModel initialised — backbone pretrained=%s, "
            "trainable params: %d",
            pretrained,
            self.count_parameters(trainable_only=True),
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the full multi-task forward pass.

        Parameters
        ----------
        x:
            Input batch of shape ``(B, 3, H, W)``.  H = W = 512 is the
            canonical training resolution, but any size compatible with the
            backbone and decoder can be used at inference.

        Returns
        -------
        dict with keys ``"heatmaps"``, ``"quality"``, ``"age"``:

        ``heatmaps``:  ``(B, 4, H, W)``  — spatial concern heatmaps
        ``quality``:   ``(B, 28)``        — zone/concern quality scores
        ``age``:       ``(B, 1)``         — biological skin-age estimate
        """
        # 1. Extract multi-scale features and global pooled representation.
        features: List[torch.Tensor]
        pooled: torch.Tensor
        features, pooled = self.backbone(x)

        # 2. Decode skip features into spatial heatmaps.
        heatmaps: torch.Tensor = self.decoder(features)

        # 3. Map pooled representation to per-zone quality scores.
        quality: torch.Tensor = self.quality_head(pooled)

        # 4. Regress a single biological age scalar.
        age: torch.Tensor = self.age_head(pooled)

        return {
            "heatmaps": heatmaps,  # (B, 4, H, W)
            "quality": quality,    # (B, 28)
            "age": age,            # (B, 1)
        }

    # ---------------------------------------------------------------------- #
    # Backbone freeze / unfreeze helpers                                       #
    # ---------------------------------------------------------------------- #

    def freeze_backbone(self) -> None:
        """Freeze backbone encoder weights for Phase-1 training.

        Only the backbone's internal encoder is frozen; the pooling head
        (conv_head / bn2 / act2) stays trainable because it directly feeds
        the downstream regression heads and benefits from gradient flow.

        Calling ``model.train()`` after this method will not inadvertently
        re-enable the frozen encoder's BatchNorm statistics, because
        ``SkinAgeBackbone.train()`` is overridden to honour the frozen state.
        """
        self.backbone.freeze()
        logger.info("Backbone encoder frozen — Phase-1 training mode.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone encoder weights for Phase-2 fine-tuning."""
        self.backbone.unfreeze()
        logger.info("Backbone encoder unfrozen — Phase-2 fine-tuning mode.")

    # ---------------------------------------------------------------------- #
    # Class-method constructors                                                #
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_config(
        cls,
        config_path: str = "SkinAge/config/model_config.yaml",
    ) -> "SkinAgeModel":
        """Construct a SkinAgeModel from a YAML configuration file.

        The YAML is expected to match the schema in
        ``SkinAge/config/model_config.yaml``.  Only the keys relevant to
        model construction are consumed here; training / optimiser keys are
        silently ignored.

        Parameters
        ----------
        config_path:
            Path to the YAML file.  Relative paths are resolved from the
            current working directory.

        Returns
        -------
        SkinAgeModel
            Fully constructed model with architecture derived from YAML.

        Example
        -------
        ::

            model = SkinAgeModel.from_config("SkinAge/config/model_config.yaml")
        """
        raw: Dict[str, Any] = _load_yaml(config_path)

        # ------------------------------------------------------------------ #
        # Map YAML schema -> flat constructor dict                             #
        # ------------------------------------------------------------------ #
        # YAML layout (see model_config.yaml):
        #   backbone.pretrained
        #   unet_decoder.output_channels          (decoder_channels not in YAML)
        #   quality_head.layers[1]                -> quality_hidden
        #   quality_head.dropout                  -> quality_dropout
        #   age_head.layers[1]                    -> age_hidden
        #   age_head.dropout                      -> age_dropout

        pretrained: bool = raw.get("backbone", {}).get("pretrained", True)

        cfg: Dict[str, Any] = {}

        unet_cfg = raw.get("unet_decoder", {})
        cfg["output_channels"] = unet_cfg.get("output_channels", 4)
        # decoder_channels is not in the YAML spec; keep the code default.

        qh_cfg = raw.get("quality_head", {})
        qh_layers: List[int] = qh_cfg.get("layers", [1408, 512, 28])
        # layers = [in, hidden, out] — we expose the hidden dimension
        cfg["quality_hidden"] = qh_layers[1] if len(qh_layers) >= 2 else 512
        cfg["quality_dropout"] = qh_cfg.get("dropout", 0.3)

        ah_cfg = raw.get("age_head", {})
        ah_layers: List[int] = ah_cfg.get("layers", [1408, 256, 1])
        cfg["age_hidden"] = ah_layers[1] if len(ah_layers) >= 2 else 256
        cfg["age_dropout"] = ah_cfg.get("dropout", 0.3)

        logger.info("Building SkinAgeModel from config: %s", config_path)
        return cls(config=cfg, pretrained=pretrained)

    # ---------------------------------------------------------------------- #
    # Parameter utilities                                                      #
    # ---------------------------------------------------------------------- #

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of model parameters.

        Parameters
        ----------
        trainable_only:
            When ``True`` (default) only count parameters with
            ``requires_grad = True``.  Set to ``False`` to count all
            parameters including frozen ones.

        Returns
        -------
        int
            Number of scalar parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ---------------------------------------------------------------------- #
    # Checkpoint helpers                                                       #
    # ---------------------------------------------------------------------- #

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer_state: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist model weights and optional training state to disk.

        The checkpoint dict contains at minimum:

        ``epoch``            — completed epoch index
        ``model_state_dict`` — ``nn.Module.state_dict()``
        ``optimizer_state``  — raw optimizer ``state_dict()`` or ``None``
        ``meta``             — caller-supplied extras (metrics, config, …)

        Parameters
        ----------
        path:
            Destination file path (usually ``*.pt`` or ``*.pth``).
        epoch:
            Index of the completed training epoch.
        optimizer_state:
            ``optimizer.state_dict()`` to enable full training resumption.
        extra_meta:
            Any JSON-serialisable metadata (e.g. val loss, config hash).
        """
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state": optimizer_state,
            "meta": extra_meta or {},
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info("Checkpoint saved to %s (epoch %d).", save_path, epoch)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        config: Optional[Dict[str, Any]] = None,
        map_location: Optional[Any] = None,
    ) -> "SkinAgeModel":
        """Restore a SkinAgeModel from a checkpoint produced by
        :meth:`save_checkpoint`.

        Parameters
        ----------
        path:
            Path to the ``.pt`` / ``.pth`` checkpoint file.
        config:
            Optional architecture config dict forwarded to ``__init__``.
            When ``None``, defaults are used (same as ``SkinAgeModel()``).
        map_location:
            Passed to ``torch.load``; use ``"cpu"`` to load a GPU checkpoint
            on a CPU-only machine.

        Returns
        -------
        SkinAgeModel
            Model with weights restored, ready for inference or resumption.
        """
        checkpoint: Dict[str, Any] = torch.load(
            path, map_location=map_location, weights_only=True
        )
        model = cls(config=config, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint from %s (epoch %d).",
            path,
            checkpoint.get("epoch", -1),
        )
        return model

    # ---------------------------------------------------------------------- #
    # Repr                                                                     #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:  # pragma: no cover
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return (
            f"SkinAgeModel("
            f"total_params={total:,}, "
            f"trainable_params={trainable:,})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = SkinAgeModel(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Heatmaps: {out['heatmaps'].shape}")   # (2, 4, 512, 512)
    print(f"Quality:  {out['quality'].shape}")    # (2, 28)
    print(f"Age:      {out['age'].shape}")        # (2, 1)
    print(f"Parameters: {model.count_parameters():,}")
    print("Full model: all checks passed!")
