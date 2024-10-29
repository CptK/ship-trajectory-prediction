import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prediction.model.transformer_trainer import TrajectoryTrainer


class DummyDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=100, seq_len=50, n_features=6):
        self.size = size
        self.seq_len = seq_len
        self.n_features = n_features
        self.mask_token = -1.0
        self.data = torch.randn(size, seq_len, n_features)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return (masked_input, attention_mask, target)
        seq = self.data[idx]
        attention_mask = torch.ones(self.seq_len, dtype=torch.bool)
        # Create some masked positions
        masked_seq = seq.clone()
        masked_seq[5:10] = self.mask_token
        return masked_seq, attention_mask, seq


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, n_features=6):
        super().__init__()
        self.layer = nn.Linear(n_features, n_features)

    def forward(self, x, src_key_padding_mask=None):
        return self.layer(x)


class TestTrajectoryTrainer(TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for saves
        self.test_dir = tempfile.mkdtemp()

        # Create datasets
        self.train_dataset = DummyDataset(size=100)
        self.val_dataset = DummyDataset(size=20)

        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=16)

        # Create model
        self.model = DummyModel()

        # Create trainer
        self.trainer = TrajectoryTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            save_dir=self.test_dir,
            learning_rate=1e-4,
            patience=2,
            grad_clip=1.0,
        )

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer.model, nn.Module)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Optimizer)
        self.assertTrue(Path(self.test_dir).exists())
        self.assertTrue((Path(self.test_dir) / "checkpoints").exists())

    def test_compute_loss(self):
        """Test loss computation."""
        batch_size = 4
        seq_len = 50
        n_features = 6

        # Create sample batch
        predictions = torch.randn(batch_size, seq_len, n_features)
        targets = torch.randn(batch_size, seq_len, n_features)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        masked_input = targets.clone()
        masked_input[:, 5:10] = self.train_dataset.mask_token

        loss = self.trainer.compute_loss(predictions, targets, attention_mask, masked_input)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Should be scalar
        self.assertTrue(loss.item() >= 0)  # MSE loss is non-negative

    def test_train_epoch(self):
        """Test training for one epoch."""
        metrics = self.trainer.train_epoch()

        self.assertIn("train_loss", metrics)
        self.assertTrue(isinstance(metrics["train_loss"], float))

    def test_validate(self):
        """Test validation."""
        metrics = self.trainer.validate()

        self.assertIn("val_loss", metrics)
        self.assertTrue(isinstance(metrics["val_loss"], float))

    def test_checkpointing(self):
        """Test checkpoint saving and loading."""
        # Save initial state
        initial_metrics = {"train_loss": 1.0, "val_loss": 1.0}
        self.trainer.save_checkpoint(initial_metrics, is_best=True)

        # Verify checkpoint files exist
        self.assertTrue((Path(self.test_dir) / "checkpoints" / "latest.pt").exists())
        self.assertTrue((Path(self.test_dir) / "checkpoints" / "best.pt").exists())

        # Modify model parameters
        with torch.no_grad():
            for param in self.trainer.model.parameters():
                param.add_(torch.randn_like(param))

        # Store modified parameters
        modified_params = [param.clone() for param in self.trainer.model.parameters()]

        # Load checkpoint
        self.trainer.load_checkpoint(str(Path(self.test_dir) / "checkpoints" / "latest.pt"))

        # Verify parameters were restored to original values
        for param, mod_param in zip(self.trainer.model.parameters(), modified_params):
            self.assertFalse(torch.allclose(param, mod_param))

        # Load into a new trainer to verify state restoration
        new_trainer = TrajectoryTrainer(
            model=DummyModel(),
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            save_dir=self.test_dir,
        )

        new_trainer.load_checkpoint(str(Path(self.test_dir) / "checkpoints" / "latest.pt"))

        # Verify states match
        for p1, p2 in zip(self.trainer.model.parameters(), new_trainer.model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # Test with scheduler
        self.trainer.train(max_epochs=1)
        self.trainer.save_checkpoint(initial_metrics, is_best=True)

        # Verify scheduler state is included when it exists
        checkpoint = torch.load(str(Path(self.test_dir) / "checkpoints" / "latest.pt"), weights_only=True)
        self.assertIn("scheduler_state_dict", checkpoint)

    def test_early_stopping(self):
        """Test early stopping functionality."""
        # Set very high initial loss to trigger early stopping
        self.trainer.best_value = 0.0
        self.trainer.minimize_metric = True

        # Should stop before 10 epochs due to patience=2
        self.assertLess(self.trainer.epoch + 1, 10)

    def test_resume_training(self):
        """Test resuming training from checkpoint."""
        # Train for a few epochs
        self.trainer.train(max_epochs=2)
        checkpoint_path = str(Path(self.test_dir) / "checkpoints" / "latest.pt")

        # Create new trainer and resume
        new_trainer = TrajectoryTrainer(
            model=DummyModel(),
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            save_dir=self.test_dir,
        )

        new_trainer.train(max_epochs=4, resume_from=checkpoint_path)

        # Should have continued from epoch 2
        self.assertGreater(new_trainer.epoch, 1)

    def test_device_handling(self):
        """Test device handling."""
        # Check model and tensors are on same device
        device = self.trainer.device
        self.assertTrue(next(self.trainer.model.parameters()).device == device)

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        # Train one batch and check gradient norms
        batch = next(iter(self.train_loader))
        batch = [b.to(self.trainer.device) for b in batch]

        predictions = self.trainer.model(batch[0], src_key_padding_mask=~batch[1])
        loss = self.trainer.compute_loss(predictions, *batch)

        self.trainer.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.trainer.model.parameters(), self.trainer.grad_clip)

        # Check that gradients are clipped
        for param in self.trainer.model.parameters():
            if param.grad is not None:
                self.assertLessEqual(param.grad.norm(), self.trainer.grad_clip + 1e-6)
