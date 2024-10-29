import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TrajectoryTrainer:
    """Trainer class for the trajectory transformer model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        patience: int = 10,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
        scheduler_min_lr: float = 1e-6,
        log_interval: int = 100,
        best_metric: str = "val_loss",
        minimize_metric: bool = True,
    ):
        """Initialize the trainer.

        Args:
            model: The transformer model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            save_dir: Directory to save checkpoints and logs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            patience: Early stopping patience (in epochs)
            grad_clip: Gradient clipping value
            device: Device to train on (default: auto-detect)
            scheduler_min_lr: Minimum learning rate for scheduler
            log_interval: How often to log metrics during training
            best_metric: Metric to use for saving best model
            minimize_metric: Whether the best metric should be minimized
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.patience = patience
        self.grad_clip = grad_clip
        self.best_metric = best_metric
        self.minimize_metric = minimize_metric
        self.scheduler_min_lr = scheduler_min_lr

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Setup device
        self.device = device or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Scheduler will be initialized in train()
        self.scheduler: CosineAnnealingLR | None = None

        # Setup logging
        self.writer = SummaryWriter(self.save_dir / "logs")

        # Initialize tracking variables
        self.epoch = 0
        self.best_value = float("inf") if minimize_metric else float("-inf")
        self.epochs_without_improvement = 0
        self.train_step = 0

        # Loss function (MSE only on real values, not padding)
        self.criterion = nn.MSELoss(reduction="none")

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked MSE loss.

        Only computes loss for:
        1. Real (non-padding) positions (using attention_mask)
        2. Masked positions (where input == mask_token)

        Args:
            predictions: Model predictions
            targets: Ground truth values
            attention_mask: Mask for padding positions
            masked_input: Input with masked positions

        Returns:
            Computed loss value
        """
        # Find masked positions (where we need to predict)
        mask_positions = masked_input == self.train_loader.dataset.mask_token  # type: ignore
        mask_positions = mask_positions.any(dim=-1)  # Combine across features

        # Combine with attention mask to only include real, masked positions
        valid_positions = attention_mask & mask_positions

        # Compute MSE loss
        loss = self.criterion(predictions, targets)

        # Average loss only over valid positions
        loss = loss[valid_positions].mean()

        return cast(torch.Tensor, loss)

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict containing training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f"Epoch {self.epoch}") as pbar:
            for batch_idx, (masked_input, attention_mask, targets) in enumerate(pbar):
                # Move batch to device
                masked_input = masked_input.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(masked_input, src_key_padding_mask=~attention_mask)

                # Compute loss
                loss = self.compute_loss(predictions, targets, attention_mask, masked_input)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Log to tensorboard
                if batch_idx % self.log_interval == 0:
                    self.writer.add_scalar("train/batch_loss", loss.item(), self.train_step)
                    self.train_step += 1

        # Compute epoch metrics
        epoch_loss /= n_batches

        return {"train_loss": epoch_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dict containing validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        n_batches = len(self.val_loader)

        for masked_input, attention_mask, targets in self.val_loader:
            # Move batch to device
            masked_input = masked_input.to(self.device)
            attention_mask = attention_mask.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(masked_input, src_key_padding_mask=~attention_mask)

            # Compute loss
            loss = self.compute_loss(predictions, targets, attention_mask, masked_input)
            val_loss += loss.item()

        val_loss /= n_batches

        return {"val_loss": val_loss}

    def save_checkpoint(self, metrics: dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            metrics: Current training metrics
            is_best: Whether this is the best model so far
        """
        if self.scheduler is None:
            raise ValueError(
                "Scheduler must be initialized before saving checkpoint (i.e. call train() first)"
            )

        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_value": self.best_value,
            "train_step": self.train_step,
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")

    def load_checkpoint(self, path: str) -> dict:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.best_value = checkpoint["best_value"]
        self.train_step = checkpoint["train_step"]

        return cast(dict, checkpoint)

    def train(self, max_epochs: int, resume_from: str | None = None) -> dict[str, float]:
        """Train the model.

        Args:
            max_epochs: Maximum number of epochs to train
            resume_from: Optional checkpoint path to resume training from

        Returns:
            Dict containing best metrics achieved during training
        """
        # Load checkpoint if specified
        start_epoch = 0
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            start_epoch = checkpoint["epoch"] + 1

        # Create scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=self.scheduler_min_lr)

        # Restore scheduler state if resuming
        if resume_from and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_metrics = {}

        for epoch in range(start_epoch, max_epochs):
            self.epoch = epoch
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            for name, value in metrics.items():
                self.writer.add_scalar(f"epoch/{name}", value, epoch)

            # Learning rate logging
            self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], epoch)

            # Check for best model
            current_value = metrics[self.best_metric]
            is_best = (self.minimize_metric and current_value < self.best_value) or (
                not self.minimize_metric and current_value > self.best_value
            )

            if is_best:
                self.best_value = current_value
                self.epochs_without_improvement = 0
                best_metrics = metrics.copy()
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(metrics, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{max_epochs} - Time: {epoch_time:.2f}s")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)

        return best_metrics
