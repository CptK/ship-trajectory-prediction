import math
from unittest import TestCase

import torch
import torch.nn as nn
from torch.testing import assert_close

from prediction.model.transformer import (
    TrajectoryDecoder,
    TrajectoryEmbedding,
    TrajectoryEncoder,
    TrajectoryTransformer,
)


class TestTrajectoryEmbedding(TestCase):
    def setUp(self):
        """Set up common test parameters and instantiate the embedding."""
        self.batch_size = 8
        self.seq_len = 50
        self.input_dim = 6
        self.d_model = 128
        self.embedding = TrajectoryEmbedding(
            input_dim=self.input_dim, d_model=self.d_model, max_seq_len=100, dropout=0.1
        )
        # Set to eval mode to disable dropout for deterministic testing
        self.embedding.eval()

    def test_initialization(self):
        """Test if the module is initialized with correct dimensions."""
        self.assertEqual(self.embedding.input_projection.in_features, self.input_dim)
        self.assertEqual(self.embedding.input_projection.out_features, self.d_model)
        self.assertEqual(self.embedding.norm.normalized_shape[0], self.d_model)

    def test_invalid_initialization(self):
        """Test if invalid initialization parameters raise appropriate errors."""
        with self.assertRaises(ValueError):
            TrajectoryEmbedding(input_dim=-1, d_model=128)

        with self.assertRaises(ValueError):
            TrajectoryEmbedding(input_dim=6, d_model=-128)

        with self.assertRaises(ValueError):
            TrajectoryEmbedding(input_dim=6, d_model=127)  # Must be even

        with self.assertRaises(ValueError):
            TrajectoryEmbedding(input_dim=6, d_model=128, max_seq_len=0)

        with self.assertRaises(ValueError):
            TrajectoryEmbedding(input_dim=6, d_model=128, dropout=1.5)

    def test_forward_shape(self):
        """Test if the output tensor has the correct shape."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.embedding(x)

        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_positional_encoding(self):
        """Test if positional encodings are being added correctly."""
        # Start by disabling layer normalization
        self.embedding.norm = nn.Identity()  # type: ignore
        # Make linear layer identity
        self.embedding.input_projection = nn.Identity()  # type: ignore
        # Disable dropout for testing
        self.embedding.dropout.p = 0

        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_clone = x.clone()  # Store original input
        output = self.embedding(x)

        # First verify that the input is preserved (identity transformation)
        assert_close(x, x_clone, msg="Input should remain unchanged")

        # Verify output is input plus positional encoding
        diff = output - x

        # Check if positional encodings are added correctly
        for i in range(self.seq_len):
            # Calculate expected positional encoding
            pos_enc = torch.zeros(self.d_model)
            for j in range(0, self.d_model, 2):
                div_term = 10000 ** (j / self.d_model)
                pos_enc[j] = math.sin(i / div_term)
                pos_enc[j + 1] = math.cos(i / div_term)

            # Compare expected positional encoding with actual difference
            # We compare with the difference because output = input + pos_enc
            assert_close(
                diff[0, i, :],  # Take first batch as they should all be the same
                pos_enc,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Positional encoding mismatch at position {i}",
            )

            # Verify that positional encodings are the same across all batches
            for batch in range(1, self.batch_size):
                assert_close(
                    diff[batch, i, :],
                    diff[0, i, :],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Positional encoding not consistent across batches at position {i}",
                )

    def test_positional_encoding_properties(self):
        """Test mathematical properties of the positional encodings."""
        # Get raw positional encodings
        pe = self.embedding.pe[: self.seq_len]

        # Test that even indices are sine and odd indices are cosine
        positions = torch.arange(self.seq_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))

        # Check sine values
        expected_sine = torch.sin(positions * div_terms)
        assert_close(pe[:, 0::2], expected_sine, rtol=1e-5, atol=1e-5, msg="Sine values mismatch")

        # Check cosine values
        expected_cosine = torch.cos(positions * div_terms)
        assert_close(pe[:, 1::2], expected_cosine, rtol=1e-5, atol=1e-5, msg="Cosine values mismatch")

    def test_masking(self):
        """Test if masking properly zeros out padded positions."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[:, self.seq_len // 2 :] = True  # Mask second half of sequence

        output = self.embedding(x, mask)

        # Check if masked positions have different embeddings
        first_half = output[:, : self.seq_len // 2, :]
        second_half = output[:, self.seq_len // 2 :, :]

        # The masked positions should have different patterns
        self.assertFalse(torch.allclose(first_half.mean(), second_half.mean()))

    def test_dropout(self):
        """Test if dropout is working as expected."""
        self.embedding.train()  # Set to training mode to enable dropout
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Run forward pass twice with same input
        out1 = self.embedding(x)
        out2 = self.embedding(x)

        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(out1, out2))

    def test_deterministic_no_dropout(self):
        """Test if output is deterministic when dropout is disabled."""
        self.embedding.eval()  # Set to eval mode to disable dropout
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Run forward pass twice with same input
        out1 = self.embedding(x)
        out2 = self.embedding(x)

        # Outputs should be identical
        assert_close(out1, out2)

    def test_long_sequence(self):
        """Test if the module handles sequences up to max_seq_len."""
        x = torch.randn(1, 100, self.input_dim)  # max_seq_len is 100
        try:
            output = self.embedding(x)
            self.assertEqual(output.shape, (1, 100, self.d_model))
        except Exception as e:
            self.fail(f"Failed to process sequence of length 100: {str(e)}")

        # Test sequence longer than max_seq_len
        x_too_long = torch.randn(1, 101, self.input_dim)
        with self.assertRaises(RuntimeError):
            _ = self.embedding(x_too_long)


class TestTrajectoryEncoder(TestCase):
    def setUp(self):
        """Initialize common test parameters and encoder instance."""
        self.batch_size = 32
        self.seq_len = 50
        self.d_model = 128
        self.nhead = 8

        self.encoder = TrajectoryEncoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=3, dim_feedforward=512, dropout=0.1
        )
        # Set to eval mode to disable dropout for deterministic testing
        self.encoder.eval()

    def test_output_shape(self):
        """Test if output maintains the expected shape."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.encoder(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_padding_mask(self):
        """Test if padding mask properly prevents attention to masked positions."""
        # Create two identical sequences except for the last 10 positions
        x1 = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x2 = x1.clone()
        x2[:, -10:, :] = torch.randn_like(x2[:, -10:, :]) * 100.0  # Very different values

        # Create padding mask for the last 10 positions
        padding_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        padding_mask[:, -10:] = True

        # Get outputs with padding mask
        output1 = self.encoder(x1, src_key_padding_mask=padding_mask)
        output2 = self.encoder(x2, src_key_padding_mask=padding_mask)

        # Check unmasked positions (they should be identical since masked positions
        # shouldn't influence them)
        assert_close(output1[:, :-10, :], output2[:, :-10, :], rtol=1e-5, atol=1e-5)

        # Optional: Verify that masked positions are different (since inputs were different)
        # Note: This is not strictly necessary for testing mask functionality
        masked_diff = (output1[:, -10:, :] - output2[:, -10:, :]).abs().mean()
        self.assertGreater(masked_diff, 0)

    def test_causal_attention(self):
        """Test if causal attention mask prevents attending to future positions."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Create causal mask (future positions are masked)
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool()

        output = self.encoder(x, src_mask=causal_mask)

        # Each position should only depend on previous positions
        # We can test this by changing future inputs and checking if output changes
        x_modified = x.clone()
        x_modified[:, -1, :] = 100.0  # Modify last position significantly

        output_modified = self.encoder(x_modified, src_mask=causal_mask)

        # First position should be identical in both outputs
        assert_close(output[:, 0, :], output_modified[:, 0, :])

    def test_batch_independence(self):
        """Test if samples in batch are processed independently."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Process full batch
        full_output = self.encoder(x)

        # Process first sample separately
        single_output = self.encoder(x[0:1])

        # First sample should be the same in both cases
        assert_close(full_output[0], single_output[0])

    def test_invalid_inputs(self):
        """Test if invalid inputs raise appropriate errors."""
        # Wrong input dimension
        with self.assertRaises(AssertionError):
            x_wrong_dim = torch.randn(self.batch_size, self.seq_len, self.d_model + 1)
            self.encoder(x_wrong_dim)

        # Wrong mask shape
        with self.assertRaises(RuntimeError):
            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            wrong_mask = torch.zeros(self.seq_len + 1, self.seq_len + 1).bool()
            self.encoder(x, src_mask=wrong_mask)

    def test_no_mask(self):
        """Test if encoder works without any masks."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.encoder(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_initialization(self):
        """Test if encoder initialization validates parameters correctly."""
        # Test invalid d_model/nhead combination
        with self.assertRaises(ValueError):
            TrajectoryEncoder(d_model=127, nhead=8, num_layers=3, dim_feedforward=512)

        # Test invalid dropout
        with self.assertRaises(ValueError):
            TrajectoryEncoder(d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=1.5)

        # Test negaive d-model
        with self.assertRaises(ValueError):
            TrajectoryEncoder(d_model=-128, nhead=8, num_layers=3, dim_feedforward=512)

        # Test negative nhead
        with self.assertRaises(ValueError):
            TrajectoryEncoder(d_model=128, nhead=-8, num_layers=3, dim_feedforward=512)

        # Test negative num_layers
        with self.assertRaises(ValueError):
            TrajectoryEncoder(d_model=128, nhead=8, num_layers=-3, dim_feedforward=512)

        # Test negative dim_feedforward
        with self.assertRaises(ValueError):
            TrajectoryEncoder(d_model=128, nhead=8, num_layers=3, dim_feedforward=-512)

    def test_sequence_length_invariance(self):
        """Test if encoder can handle different sequence lengths."""
        x_short = torch.randn(self.batch_size, 25, self.d_model)
        x_long = torch.randn(self.batch_size, 100, self.d_model)

        output_short = self.encoder(x_short)
        output_long = self.encoder(x_long)

        self.assertEqual(output_short.shape, (self.batch_size, 25, self.d_model))
        self.assertEqual(output_long.shape, (self.batch_size, 100, self.d_model))

    def test_gradient_flow(self):
        """Test if gradients can flow through the encoder."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x.requires_grad = True

        output = self.encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())  # type: ignore


class TestTrajectoryDecoder(TestCase):
    """Test suite for the TrajectoryDecoder class."""

    def setUp(self):
        """Set up common parameters and instances for tests."""
        self.batch_size = 32
        self.seq_len = 10
        self.memory_len = 20
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 3
        self.dim_feedforward = 1024

        # Create decoder instance
        self.decoder = TrajectoryDecoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
        )

        # Set to eval mode to disable dropout for deterministic testing
        self.decoder.eval()

    def test_initialization(self):
        """Test that decoder initializes with valid parameters."""
        decoder = TrajectoryDecoder(d_model=256, nhead=8, num_layers=3, dim_feedforward=1024)
        self.assertIsInstance(decoder.decoder, nn.TransformerDecoder)

    def test_invalid_initialization(self):
        """Test that decoder raises errors for invalid parameters."""
        # Test invalid d_model/nhead combination
        with self.assertRaises(ValueError):
            TrajectoryDecoder(
                d_model=255, nhead=8, num_layers=3, dim_feedforward=1024  # Not divisible by nhead
            )

        # Test negative dimensions
        with self.assertRaises(ValueError):
            TrajectoryDecoder(d_model=-256, nhead=8, num_layers=3, dim_feedforward=1024)

        # Test invalid dropout
        with self.assertRaises(ValueError):
            TrajectoryDecoder(d_model=256, nhead=8, num_layers=3, dim_feedforward=1024, dropout=1.5)

    def test_forward_shape(self):
        """Test that output shapes are correct."""
        tgt = torch.randn(self.batch_size, self.seq_len, self.d_model)
        memory = torch.randn(self.batch_size, self.memory_len, self.d_model)

        output = self.decoder(tgt, memory)

        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_mask(self):
        """Test forward pass with various mask configurations."""
        tgt = torch.randn(self.batch_size, self.seq_len, self.d_model)
        memory = torch.randn(self.batch_size, self.memory_len, self.d_model)

        # Create causal mask for target sequence
        # Convert to same dtype as padding mask (bool)
        tgt_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool()

        # Create padding masks
        tgt_padding_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        tgt_padding_mask[:, -2:] = True  # Mask last two positions

        memory_padding_mask = torch.zeros(self.batch_size, self.memory_len, dtype=torch.bool)
        memory_padding_mask[:, -2:] = True  # Mask last two positions

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

        # Additional assertions to verify masking behavior
        # Check that masked positions have different patterns than unmasked ones
        # We'll look at the mean activation in masked vs unmasked positions
        masked_output = output[:, -2:, :]  # Last two positions were masked
        unmasked_output = output[:, :-2, :]

        masked_mean = masked_output.mean()
        unmasked_mean = unmasked_output.mean()

        # The masked and unmasked positions should have different patterns
        # but we can't know exactly what they'll be, so we just check they differ
        self.assertNotEqual(masked_mean.item(), unmasked_mean.item())

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic in eval mode."""
        tgt = torch.randn(self.batch_size, self.seq_len, self.d_model)
        memory = torch.randn(self.batch_size, self.memory_len, self.d_model)

        # Get outputs from two forward passes
        output1 = self.decoder(tgt, memory)
        output2 = self.decoder(tgt, memory)

        # Verify outputs are identical
        assert_close(output1, output2)

    def test_device_transfer(self):
        """Test that decoder can be moved to GPU if available."""
        if not torch.cuda.is_available() and not torch.mps.is_available():
            self.skipTest("Neither CUDA nor MPS are available")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        self.decoder.to(device)

        tgt = torch.randn(self.batch_size, self.seq_len, self.d_model, device=device)
        memory = torch.randn(self.batch_size, self.memory_len, self.d_model, device=device)

        output = self.decoder(tgt, memory)

        # Compare device types instead of full device objects
        self.assertEqual(output.device.type, device.type)

        # Additional checks to ensure proper device placement
        self.assertTrue(output.is_cuda if torch.cuda.is_available() else output.is_mps)

        # Verify all model parameters are on the correct device
        for param in self.decoder.parameters():
            self.assertEqual(param.device.type, device.type)

    def test_gradient_flow(self):
        """Test that gradients flow through the decoder."""
        self.decoder.train()  # Set to training mode

        tgt = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        memory = torch.randn(self.batch_size, self.memory_len, self.d_model, requires_grad=True)

        output = self.decoder(tgt, memory)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(tgt.grad)
        self.assertIsNotNone(memory.grad)

        # Check that model parameters have gradients
        for param in self.decoder.parameters():
            self.assertIsNotNone(param.grad)

    def test_long_sequence(self):
        """Test decoder with longer sequences."""
        long_seq_len = 100
        tgt = torch.randn(self.batch_size, long_seq_len, self.d_model)
        memory = torch.randn(self.batch_size, self.memory_len, self.d_model)

        output = self.decoder(tgt, memory)
        expected_shape = (self.batch_size, long_seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_small_batch(self):
        """Test decoder with batch size of 1."""
        small_batch = 1
        tgt = torch.randn(small_batch, self.seq_len, self.d_model)
        memory = torch.randn(small_batch, self.memory_len, self.d_model)

        output = self.decoder(tgt, memory)
        expected_shape = (small_batch, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)


class TestTrajectoryTransformer(TestCase):
    """Test suite for the TrajectoryTransformer class."""

    def setUp(self):
        """Set up common parameters and instances for tests."""
        self.batch_size = 32
        self.seq_len = 50
        self.d_model = 256
        self.nhead = 8
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dim_feedforward = 1024
        self.dropout = 0.1

        # Create basic transformer instance with all features
        self.transformer = TrajectoryTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            include_speed=True,
            include_orientation=True,
            include_timestamps=True,
        )

        # Set to eval mode for deterministic testing
        self.transformer.eval()

    def test_initialization_all_features(self):
        """Test transformer initialization with all features enabled."""
        transformer = TrajectoryTransformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=1024,
            include_speed=True,
            include_orientation=True,
            include_timestamps=True,
        )

        # Check input dimension calculation
        self.assertEqual(transformer.input_dim, 6)  # 2 (lat,lon) + 1 (speed) + 2 (orientation) + 1 (time)

        # Check component initialization
        self.assertIsInstance(transformer.embedding, TrajectoryEmbedding)
        self.assertIsInstance(transformer.encoder, TrajectoryEncoder)
        self.assertIsInstance(transformer.decoder, TrajectoryDecoder)
        self.assertIsInstance(transformer.output_projection, nn.Linear)

    def test_initialization_minimal_features(self):
        """Test transformer initialization with minimal features."""
        transformer = TrajectoryTransformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=1024,
            include_speed=False,
            include_orientation=False,
            include_timestamps=False,
        )

        self.assertEqual(transformer.input_dim, 2)  # Only lat, lon
        self.assertEqual(transformer.output_projection.out_features, 2)

    def test_forward_shape(self):
        """Test that output shapes are correct for various input sizes."""
        # Test with different sequence lengths
        for seq_len in [10, 50, 100]:
            with self.subTest(seq_length=seq_len):
                src = torch.randn(self.batch_size, seq_len, self.transformer.input_dim)
                output = self.transformer(src)

                expected_shape = (self.batch_size, seq_len, self.transformer.input_dim)
                self.assertEqual(output.shape, expected_shape)

    def test_forward_with_padding_mask(self):
        """Test forward pass with padding mask."""
        src = torch.randn(self.batch_size, self.seq_len, self.transformer.input_dim)

        # Create padding mask (mask last 10 positions)
        padding_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        padding_mask[:, -10:] = True

        output = self.transformer(src, padding_mask)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.transformer.input_dim)
        self.assertEqual(output.shape, expected_shape)

        # Verify masked positions have different patterns
        masked_output = output[:, -10:, :]
        unmasked_output = output[:, :-10, :]

        self.assertNotEqual(masked_output.mean().item(), unmasked_output.mean().item())

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic in eval mode."""
        src = torch.randn(self.batch_size, self.seq_len, self.transformer.input_dim)

        # Get outputs from two forward passes
        output1 = self.transformer(src)
        output2 = self.transformer(src)

        # Verify outputs are identical
        assert_close(output1, output2)

    def test_gradient_flow(self):
        """Test that gradients flow through all components."""
        self.transformer.train()  # Set to training mode

        src = torch.randn(self.batch_size, self.seq_len, self.transformer.input_dim, requires_grad=True)

        output = self.transformer(src)
        loss = output.sum()
        loss.backward()

        # Check that input gradients are computed
        self.assertIsNotNone(src.grad)

        # Check that all model components have gradients
        for name, param in self.transformer.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))  # type: ignore

    def test_device_transfer(self):
        """Test that model can be moved to available devices."""
        if not torch.cuda.is_available() and not torch.mps.is_available():
            self.skipTest("Neither CUDA nor MPS available")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        self.transformer.to(device)

        src = torch.randn(self.batch_size, self.seq_len, self.transformer.input_dim, device=device)

        output = self.transformer(src)

        # Check output device
        self.assertEqual(output.device.type, device.type)

        # Check all parameters are on correct device
        for param in self.transformer.parameters():
            self.assertEqual(param.device.type, device.type)

    def test_invalid_input_shape(self):
        """Test that invalid input shapes raise appropriate errors."""
        # Wrong feature dimension
        wrong_features = torch.randn(self.batch_size, self.seq_len, self.transformer.input_dim + 1)
        with self.assertRaises(RuntimeError):
            self.transformer(wrong_features)

        # Wrong number of dimensions
        wrong_dims = torch.randn(self.batch_size, self.seq_len)
        with self.assertRaises(RuntimeError):
            self.transformer(wrong_dims)

    def test_long_sequence(self):
        """Test model with longer sequences."""
        long_seq_len = 200
        src = torch.randn(self.batch_size, long_seq_len, self.transformer.input_dim)
        output = self.transformer(src)

        expected_shape = (self.batch_size, long_seq_len, self.transformer.input_dim)
        self.assertEqual(output.shape, expected_shape)
