"""
Two-phase training loop for the SkinAge multi-task model.

Training strategy
-----------------
Phase 1 — Backbone frozen (3 epochs, LR 1e-3)
    Only task heads receive gradients.  The EfficientNet-B2 encoder runs in
    eval mode so its BatchNorm running statistics are not corrupted before
    fine-tuning begins.  AdamW optimises only the unfrozen parameters.

Phase 2 — Full fine-tune (up to 30 epochs, LR 5e-5)
    Backbone is unfrozen.  A *new* AdamW optimizer is constructed so momentum
    buffers do not carry stale Phase-1 values into Phase-2.  CosineAnnealingLR
    decays the learning rate to eta_min=1e-6 over T_max epochs.  EarlyStopping
    halts training if the composite validation loss fails to improve for 7
    consecutive epochs.

Mixed-label batches
-------------------
The DataLoader (via skinage_collate_fn) assembles batches that naturally
contain both age-labelled (UTKFace) and age-unlabelled (FFHQ/CelebA) samples.
The ``age_indices`` LongTensor in each batch selects the subset of predictions
that have ground-truth age labels before computing the age regression loss.
The trainer does not manipulate batch composition; it relies entirely on the
collate function.

Mixed precision
---------------
torch.amp.autocast + GradScaler are used for all CUDA training to avoid OOM
errors with 512x512 images at batch size 16.  On CPU the GradScaler is a
no-op (enabled=False) to keep the same code path.

Checkpointing
-------------
The best validation composite loss triggers a checkpoint save under
``{output_dir}/best_model.pth``.  A final checkpoint is written at the end
of training as ``{output_dir}/final_model.pth``.

History
-------
Per-epoch metrics are accumulated in ``self.history`` and serialised to
``{output_dir}/training_history.json`` for downstream notebook analysis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.utils.data
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitor a validation metric and signal when training should stop.

    The caller passes the current validation loss on every epoch.  The
    instance returns ``True`` when the loss improved (caller should save a
    checkpoint) and ``False`` otherwise.  After ``patience`` consecutive
    epochs without improvement the ``should_stop`` flag is set to ``True``.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training is stopped.
    min_delta : float
        Minimum absolute change in the monitored metric to qualify as an
        improvement.  Defaults to ``0.0`` (any decrease counts).

    Attributes
    ----------
    best_loss : float | None
        Best validation loss seen so far.
    counter : int
        Number of consecutive epochs without improvement.
    should_stop : bool
        Set to ``True`` when the patience threshold is reached.

    Example
    -------
    >>> es = EarlyStopping(patience=3)
    >>> es(1.0)   # improved, returns True
    True
    >>> es(0.8)   # improved, returns True
    True
    >>> es(0.85)  # no improvement, returns False
    False
    >>> es.should_stop
    False
    >>> es(0.85)
    False
    >>> es(0.85)
    False
    >>> es.should_stop
    True
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: Optional[float] = None
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Evaluate the current validation loss.

        Parameters
        ----------
        val_loss : float
            Current epoch validation composite loss.

        Returns
        -------
        bool
            ``True`` if the metric improved (caller should save checkpoint),
            ``False`` otherwise.
        """
        if self.best_loss is None:
            # First call — always an improvement.
            self.best_loss = val_loss
            return True

        if val_loss < self.best_loss - self.min_delta:
            # Genuine improvement.
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            # No improvement this epoch.
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EarlyStopping("
            f"patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"counter={self.counter}, "
            f"best_loss={self.best_loss}, "
            f"should_stop={self.should_stop})"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SkinAgeTrainer:
    """Two-phase training loop for the SkinAge multi-task model.

    Encapsulates the complete training procedure including Phase 1 (frozen
    backbone, head warm-up) and Phase 2 (full fine-tune with cosine annealing
    and early stopping), with mixed-precision support, best-checkpoint saving,
    and JSON history logging.

    Parameters
    ----------
    model : SkinAgeModel
        The assembled multi-task model.  Must expose ``freeze_backbone()``,
        ``unfreeze_backbone()``, and the standard ``nn.Module`` interface.
    criterion : MultiTaskLoss
        Multi-task loss function.  Its ``forward`` method must accept the
        ``predictions`` dict and ``targets`` dict produced by the DataLoader /
        collate function.
    config : dict
        Full ``model_config.yaml`` dict loaded with PyYAML.  Expected
        top-level keys: ``training``, ``optimizer``, ``early_stopping``,
        ``dataloader``.
    device : torch.device
        Target compute device.
    output_dir : str
        Directory for checkpoints, logs, and history JSON.  Created if it
        does not exist.  Defaults to ``"outputs"``.
    """

    # Name used for progress bar display
    _TQDM_NCOLS: int = 100

    def __init__(
        self,
        model: Any,  # SkinAgeModel — typed as Any to avoid circular import at module level
        criterion: Any,  # MultiTaskLoss
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str = "outputs",
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision: GradScaler is a no-op on CPU (enabled=False).
        self._use_amp: bool = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        # History stored as a flat list of per-epoch dicts.
        self.history: List[Dict[str, Any]] = []

        # Best validation loss seen across both phases.
        self.best_val_loss: Optional[float] = None

        # Track total epochs completed for checkpoint metadata.
        self._global_epoch: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        """Run Phase 1 followed by Phase 2 and return the training history.

        Parameters
        ----------
        train_loader : DataLoader
            Training DataLoader produced by ``build_dataloader(..., shuffle=True)``.
        val_loader : DataLoader
            Validation DataLoader produced by ``build_dataloader(..., shuffle=False)``.

        Returns
        -------
        dict
            The accumulated ``self.history`` list (one entry per epoch across
            both phases).
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("Starting SkinAge training — two-phase strategy")
        logger.info("Device: %s", self.device)
        logger.info("Mixed precision (AMP): %s", self._use_amp)
        logger.info("Output directory: %s", self.output_dir)
        logger.info("=" * 60)

        self._run_phase1(train_loader, val_loader)
        self._run_phase2(train_loader, val_loader)

        elapsed = time.time() - start
        logger.info("Training complete in %.1f minutes.", elapsed / 60.0)

        self.save_history(str(self.output_dir / "training_history.json"))
        return {"history": self.history, "best_val_loss": self.best_val_loss}

    def save_history(self, path: str) -> None:
        """Serialise the training history list to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path.  Parent directories are created if needed.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        logger.info("Training history saved to %s", dest)

    # ------------------------------------------------------------------
    # Phase runners
    # ------------------------------------------------------------------

    def _run_phase1(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> None:
        """Phase 1: frozen backbone, head warm-up."""
        phase_cfg = self.config["training"]["phase1"]
        opt_cfg = self.config["optimizer"]
        n_epochs: int = phase_cfg["epochs"]
        lr: float = phase_cfg["learning_rate"]

        logger.info("-" * 60)
        logger.info("Phase 1 — Frozen backbone, %d epochs, LR=%.1e", n_epochs, lr)

        self.model.freeze_backbone()

        # Filter parameters: only heads have requires_grad=True while frozen.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=opt_cfg["weight_decay"],
        )

        n_trainable = sum(p.numel() for p in trainable_params)
        logger.info("Trainable parameters in Phase 1: %s", f"{n_trainable:,}")

        for epoch in range(n_epochs):
            train_losses = self._train_epoch(
                train_loader, optimizer, epoch=epoch, phase="Phase1"
            )
            val_losses = self._validate_epoch(val_loader, epoch=epoch, phase="Phase1")
            self._record_epoch(
                phase="Phase1",
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                lr=optimizer.param_groups[0]["lr"],
            )
            self._global_epoch += 1

    def _run_phase2(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> None:
        """Phase 2: full fine-tune with cosine annealing and early stopping."""
        phase_cfg = self.config["training"]["phase2"]
        opt_cfg = self.config["optimizer"]
        es_cfg = self.config["early_stopping"]
        n_epochs: int = phase_cfg["epochs"]
        lr: float = phase_cfg["learning_rate"]
        eta_min: float = phase_cfg["lr_scheduler"]["eta_min"]

        logger.info("-" * 60)
        logger.info(
            "Phase 2 — Full fine-tune, up to %d epochs, LR=%.1e → %.1e",
            n_epochs, lr, eta_min,
        )

        self.model.unfreeze_backbone()

        # Fresh optimizer — new momentum buffers for all parameters.
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=opt_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=eta_min,
        )
        early_stopping = EarlyStopping(patience=es_cfg["patience"])

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Trainable parameters in Phase 2: %s", f"{n_trainable:,}")

        # Sentinel for the final checkpoint call in case the loop never executes.
        last_val_total: float = float("inf")

        for epoch in range(n_epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            train_losses = self._train_epoch(
                train_loader, optimizer, epoch=epoch, phase="Phase2"
            )
            val_losses = self._validate_epoch(val_loader, epoch=epoch, phase="Phase2")

            val_total: float = val_losses["total"]
            last_val_total = val_total
            scheduler.step()

            improved = early_stopping(val_total)

            # Save best checkpoint whenever validation loss improves.
            if improved:
                self.best_val_loss = val_total
                self._save_checkpoint(
                    epoch=self._global_epoch,
                    val_loss=val_total,
                    optimizer=optimizer,
                    filename="best_model.pth",
                )
                logger.info(
                    "  Checkpoint saved — new best val loss: %.6f", val_total
                )

            self._record_epoch(
                phase="Phase2",
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                lr=current_lr,
            )
            self._global_epoch += 1

            if early_stopping.should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d consecutive epochs).",
                    epoch, es_cfg["patience"],
                )
                break

        # Save final model regardless of whether it is also the best.
        self._save_checkpoint(
            epoch=self._global_epoch - 1,
            val_loss=last_val_total,
            optimizer=optimizer,
            filename="final_model.pth",
        )

    # ------------------------------------------------------------------
    # Inner loop helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        phase: str,
    ) -> Dict[str, float]:
        """Run one training epoch and return averaged per-head losses.

        Parameters
        ----------
        loader : DataLoader
            Training data loader.
        optimizer : torch.optim.Optimizer
            The current phase optimizer.
        epoch : int
            Zero-indexed epoch number within the current phase.
        phase : str
            Display label (``"Phase1"`` or ``"Phase2"``).

        Returns
        -------
        dict
            Keys: ``total``, ``heatmap``, ``quality``, ``age``.
            Values: epoch-averaged scalar floats.
        """
        self.model.train()
        device_type = self.device.type

        accum: Dict[str, float] = {
            "total": 0.0, "heatmap": 0.0, "quality": 0.0, "age": 0.0
        }

        pbar = tqdm(
            loader,
            desc=f"[{phase}] Epoch {epoch + 1:03d} train",
            ncols=self._TQDM_NCOLS,
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            images: torch.Tensor = batch["image"].to(self.device, non_blocking=True)
            gt_heatmaps: torch.Tensor = batch["heatmaps"].to(self.device, non_blocking=True)
            gt_quality: torch.Tensor = batch["quality_scores"].to(self.device, non_blocking=True)
            gt_age: Optional[torch.Tensor] = (
                batch["age"].to(self.device, non_blocking=True)
                if batch["age"] is not None
                else None
            )
            age_indices: torch.Tensor = batch["age_indices"].to(self.device, non_blocking=True)

            # Log mixed batch composition once at the very first batch.
            if batch_idx == 0 and epoch == 0:
                age_count = age_indices.numel()
                batch_size = images.shape[0]
                logger.info(
                    "[%s] Batch composition: %d/%d samples have age labels",
                    phase, age_count, batch_size,
                )
                print(
                    f"Batch composition: {age_count}/{batch_size} samples have age labels"
                )

            # Forward pass under mixed precision context.
            with torch.amp.autocast(device_type=device_type, enabled=self._use_amp):
                predictions = self.model(images)
                losses = self.criterion(
                    predictions,
                    {
                        "heatmaps": gt_heatmaps,
                        "quality_scores": gt_quality,
                        "age": gt_age,
                        "age_indices": age_indices,
                    },
                )

            total_loss: torch.Tensor = losses["total"]

            optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # Accumulate detached scalars.
            accum["total"] += total_loss.item()
            accum["heatmap"] += losses["heatmap"].item()
            accum["quality"] += losses["quality"].item()
            accum["age"] += losses["age"].item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                hm=f"{losses['heatmap'].item():.4f}",
                q=f"{losses['quality'].item():.4f}",
                age=f"{losses['age'].item():.4f}",
            )

        n_batches = max(len(loader), 1)
        return {k: v / n_batches for k, v in accum.items()}

    def _validate_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        phase: str,
    ) -> Dict[str, float]:
        """Run one validation epoch and return averaged per-head losses.

        Parameters
        ----------
        loader : DataLoader
            Validation data loader.
        epoch : int
            Zero-indexed epoch number within the current phase.
        phase : str
            Display label (``"Phase1"`` or ``"Phase2"``).

        Returns
        -------
        dict
            Keys: ``total``, ``heatmap``, ``quality``, ``age``.
        """
        self.model.eval()
        device_type = self.device.type

        accum: Dict[str, float] = {
            "total": 0.0, "heatmap": 0.0, "quality": 0.0, "age": 0.0
        }

        pbar = tqdm(
            loader,
            desc=f"[{phase}] Epoch {epoch + 1:03d}   val",
            ncols=self._TQDM_NCOLS,
            leave=False,
        )

        with torch.no_grad():
            for batch in pbar:
                images: torch.Tensor = batch["image"].to(self.device, non_blocking=True)
                gt_heatmaps: torch.Tensor = batch["heatmaps"].to(self.device, non_blocking=True)
                gt_quality: torch.Tensor = batch["quality_scores"].to(self.device, non_blocking=True)
                gt_age: Optional[torch.Tensor] = (
                    batch["age"].to(self.device, non_blocking=True)
                    if batch["age"] is not None
                    else None
                )
                age_indices: torch.Tensor = batch["age_indices"].to(
                    self.device, non_blocking=True
                )

                with torch.amp.autocast(device_type=device_type, enabled=self._use_amp):
                    predictions = self.model(images)
                    losses = self.criterion(
                        predictions,
                        {
                            "heatmaps": gt_heatmaps,
                            "quality_scores": gt_quality,
                            "age": gt_age,
                            "age_indices": age_indices,
                        },
                    )

                accum["total"] += losses["total"].item()
                accum["heatmap"] += losses["heatmap"].item()
                accum["quality"] += losses["quality"].item()
                accum["age"] += losses["age"].item()

                pbar.set_postfix(
                    val_loss=f"{losses['total'].item():.4f}",
                )

        n_batches = max(len(loader), 1)
        return {k: v / n_batches for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Logging and history
    # ------------------------------------------------------------------

    def _record_epoch(
        self,
        phase: str,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        lr: float,
    ) -> None:
        """Append a history entry and print a formatted epoch summary.

        Parameters
        ----------
        phase : str
            ``"Phase1"`` or ``"Phase2"``.
        epoch : int
            Zero-indexed epoch within the phase.
        train_losses : dict
            Training loss components from ``_train_epoch``.
        val_losses : dict
            Validation loss components from ``_validate_epoch``.
        lr : float
            Current learning rate (first param group).
        """
        record: Dict[str, Any] = {
            "phase": phase,
            "epoch": epoch,
            "global_epoch": self._global_epoch,
            "lr": lr,
            "train_total": train_losses["total"],
            "train_heatmap": train_losses["heatmap"],
            "train_quality": train_losses["quality"],
            "train_age": train_losses["age"],
            "val_total": val_losses["total"],
            "val_heatmap": val_losses["heatmap"],
            "val_quality": val_losses["quality"],
            "val_age": val_losses["age"],
        }
        self.history.append(record)

        logger.info(
            "[%s] Epoch %03d | LR %.2e "
            "| Train: total=%.4f heatmap=%.4f quality=%.4f age=%.4f "
            "| Val:   total=%.4f heatmap=%.4f quality=%.4f age=%.4f",
            phase, epoch + 1, lr,
            train_losses["total"],
            train_losses["heatmap"],
            train_losses["quality"],
            train_losses["age"],
            val_losses["total"],
            val_losses["heatmap"],
            val_losses["quality"],
            val_losses["age"],
        )
        print(
            f"[{phase}] Epoch {epoch + 1:03d} | LR {lr:.2e} "
            f"| Train total={train_losses['total']:.4f}  "
            f"hm={train_losses['heatmap']:.4f}  "
            f"q={train_losses['quality']:.4f}  "
            f"age={train_losses['age']:.4f} "
            f"| Val total={val_losses['total']:.4f}  "
            f"hm={val_losses['heatmap']:.4f}  "
            f"q={val_losses['quality']:.4f}  "
            f"age={val_losses['age']:.4f}"
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        optimizer: torch.optim.Optimizer,
        filename: str,
    ) -> None:
        """Persist model weights and training metadata to disk.

        The checkpoint dict contains everything needed to resume training or
        perform offline evaluation:

        - ``model_state_dict`` — model weights.
        - ``optimizer_state_dict`` — optimizer state (momentum buffers, etc.).
        - ``scaler_state_dict`` — GradScaler state (for AMP resumption).
        - ``epoch`` — global epoch at the time of saving.
        - ``val_loss`` — validation composite loss for this checkpoint.
        - ``config`` — full model_config.yaml dict for reproducibility.
        - ``history`` — snapshot of training history up to this checkpoint.

        Parameters
        ----------
        epoch : int
            Global epoch index at time of saving.
        val_loss : float
            Validation composite loss for this checkpoint.
        optimizer : torch.optim.Optimizer
            Current optimizer (state saved for resumption).
        filename : str
            File name under ``self.output_dir``.
        """
        dest = self.output_dir / filename
        payload: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.config,
            "history": self.history,
        }
        torch.save(payload, dest)
        logger.info("Checkpoint saved: %s  (epoch=%d, val_loss=%.6f)", dest, epoch, val_loss)
