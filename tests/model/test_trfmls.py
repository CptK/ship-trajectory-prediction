from unittest import TestCase

import torch

from prediction.model.trfmls import TRFMLS


class TestTRFMLS(TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            "input_dim": 6,
            "d_model": 256,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 1024,
            "max_seq_len": 50,
            "output_dim": 6,
            "dropout": 0.1,
            "batch_first": True,
        }
        self.model = TRFMLS(**self.config).to(self.device)

    def test_mask_generation(self):
        """Test if the generated mask correctly prevents looking ahead."""
        seq_len = 5
        mask = self.model._generate_square_subsequent_mask(seq_len)

        # Check shape
        self.assertEqual(mask.shape, (seq_len, seq_len))

        # Check if mask is upper triangular
        expected_mask = torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
        expected_mask = expected_mask.float().masked_fill(expected_mask == 0, float("-inf"))
        expected_mask = expected_mask.masked_fill(expected_mask == 1, 0.0)

        self.assertTrue(torch.allclose(mask, expected_mask))

    def test_information_leakage(self):
        """Test if future information is properly masked in decoder."""
        batch_size = 2
        src_len = 10
        tgt_len = 5  # Make target length different from source length

        # Create input with distinct values for easy tracking
        src = torch.randn(batch_size, src_len, self.config["input_dim"]).to(self.device)

        # Create target starting with last point from src
        # Then add sequential values
        tgt = torch.zeros(batch_size, tgt_len + 1, self.config["input_dim"]).to(self.device)
        tgt[:, 0, :] = src[:, -1, :]  # Last point from source
        # Fill rest with sequential values
        seq_values = torch.arange(1, tgt_len + 1).float().unsqueeze(-1)
        tgt[:, 1:, :] = seq_values.repeat(1, self.config["input_dim"])

        # Get model output
        output = self.model(src, tgt)

        # For each position after the first (which is the known last point from src)
        with torch.no_grad():
            for i in range(2, tgt_len + 1):  # Start from 2 since pos 1 is from src
                # Create modified target with different future values
                modified_tgt = tgt.clone()
                modified_tgt[:, i:] = 999  # Change future values

                modified_output = self.model(src, modified_tgt)

                # Print detailed information about the outputs
                print(f"\nPosition {i}:")
                print(f"Original output[:, :i]:\n{output[:, :i]}")
                print(f"Modified output[:, :i]:\n{modified_output[:, :i]}")
                print(f"Difference:\n{(output[:, :i] - modified_output[:, :i]).abs().max()}")

                # Check if outputs match up to position i
                self.assertTrue(
                    torch.allclose(output[:, :i], modified_output[:, :i], atol=1e-6),
                    f"Information leak detected at position {i}",
                )

    def test_encoder_outputs(self):
        """Test if encoder outputs maintain the expected dimensions and properties."""
        batch_size = 2
        seq_len = 10
        src = torch.randn(batch_size, seq_len, self.config["input_dim"]).to(self.device)

        # First project input to d_model dimensions using model's input projection
        src_projected = self.model.input_proj(src)

        # Test single encoder layer
        encoder_layer = self.model.encoder_layers[0]
        encoder_output = encoder_layer(src_projected)

        # Check output shape
        self.assertEqual(encoder_output.shape, (batch_size, seq_len, self.config["d_model"]))

        # Check if LSTM and transformer fusion works
        # Output should be different from both pure LSTM and pure transformer
        with torch.no_grad():
            lstm_out, _ = encoder_layer.lstm(encoder_layer.input_embedding(src_projected))
            transformer_out = encoder_layer.transformer_encoder_layer(
                encoder_layer.input_embedding(src_projected)
            )

            self.assertFalse(torch.allclose(encoder_output, lstm_out, atol=1e-6))
            self.assertFalse(torch.allclose(encoder_output, transformer_out, atol=1e-6))

    def test_autoregressive_generation(self):
        """Test if model can generate outputs autoregressively."""
        batch_size = 2
        src_len = 10
        tgt_len = 5

        src = torch.randn(batch_size, src_len, self.config["input_dim"]).to(self.device)

        # Test with zero tensor as initial input
        zeros = torch.zeros(batch_size, tgt_len, self.config["input_dim"]).to(self.device)
        output = self.model(src, zeros)

        self.assertEqual(output.shape, (batch_size, tgt_len, self.config["output_dim"]))

        # Test if outputs are reasonable (not all zeros or inf)
        self.assertFalse(torch.all(output == 0))
        self.assertFalse(torch.any(torch.isinf(output)))
        self.assertFalse(torch.any(torch.isnan(output)))

    def test_positional_encoding(self):
        """Test if positional encoding is properly applied in encoder layers."""
        batch_size = 2
        seq_len = 10
        src = torch.randn(batch_size, seq_len, self.config["input_dim"]).to(self.device)

        # Project input to d_model dimensions first
        src_projected = self.model.input_proj(src)

        encoder_layer = self.model.encoder_layers[0]

        # Get output with normal sequence
        output1 = encoder_layer(src_projected)

        # Get output with shifted sequence
        shifted_src = torch.roll(src_projected, shifts=1, dims=1)
        output2 = encoder_layer(shifted_src)

        # Outputs should be different due to positional encoding
        self.assertFalse(torch.allclose(output1, output2, atol=1e-6))

    def test_gradient_flow(self):
        """Test if gradients flow through all parts of the model."""
        batch_size = 2
        seq_len = 10

        src = torch.randn(batch_size, seq_len, self.config["input_dim"]).to(self.device)
        tgt = torch.randn(batch_size, seq_len, self.config["input_dim"]).to(self.device)

        output = self.model(src, tgt)
        loss = output.mean()
        loss.backward()

        # Check if all parameters have gradients
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")

    def test_mask_values(self):
        """Test if the attention mask has correct values for causal attention."""
        seq_len = 4
        mask = self.model._generate_square_subsequent_mask(seq_len)

        # Expected mask for seq_len=4 should look like:
        # [[ 0., -inf, -inf, -inf],
        #  [ 0.,  0.,  -inf, -inf],
        #  [ 0.,  0.,   0., -inf],
        #  [ 0.,  0.,   0.,  0.]]

        expected_mask = torch.zeros(seq_len, seq_len)
        expected_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        expected_mask = (
            expected_mask.float().masked_fill(expected_mask, float("-inf")).masked_fill(~expected_mask, 0.0)
        )

        print("Generated mask:")
        print(mask)
        print("\nExpected mask:")
        print(expected_mask)

        self.assertTrue(torch.allclose(mask, expected_mask, equal_nan=True))

    def test_decoder_causality(self):
        """Test if decoder respects causality independently of encoder."""
        batch_size = 2
        seq_len = 10

        # Create fixed encoder output
        encoder_output = torch.ones(batch_size, seq_len, self.config["d_model"]).to(self.device)

        # Create sequential target with more distinct values
        tgt = (
            torch.arange(seq_len)
            .float()
            .unsqueeze(-1)
            .repeat(batch_size, 1, self.config["d_model"])
            .to(self.device)
        )

        # Create mask
        tgt_mask = self.model._generate_square_subsequent_mask(seq_len).to(self.device)

        # Test single decoder layer
        decoder_layer = self.model.decoder_layers[0]

        # Add hooks to get intermediate outputs
        self_attn_output = []
        cross_attn_output = []

        def self_attn_hook(module, input, output):
            self_attn_output.append(output[0].detach().cpu())  # Get attention output

        def cross_attn_hook(module, input, output):
            cross_attn_output.append(output[0].detach().cpu())  # Get attention output

        # Register hooks
        self_attn_hook_handle = decoder_layer.self_attn.register_forward_hook(self_attn_hook)
        cross_attn_hook_handle = decoder_layer.multihead_attn.register_forward_hook(cross_attn_hook)

        # Get output with original target
        output1 = decoder_layer(tgt, encoder_output, tgt_mask=tgt_mask)
        orig_self_attn = self_attn_output[-1]
        orig_cross_attn = cross_attn_output[-1]

        # Clear stored outputs
        self_attn_output.clear()
        cross_attn_output.clear()

        # Modify future values
        modified_tgt = tgt.clone()
        modified_tgt[:, 5:] = 999

        # Get output with modified target
        output2 = decoder_layer(modified_tgt, encoder_output, tgt_mask=tgt_mask)
        mod_self_attn = self_attn_output[-1]
        mod_cross_attn = cross_attn_output[-1]

        # Remove hooks
        self_attn_hook_handle.remove()
        cross_attn_hook_handle.remove()

        # Print detailed comparisons
        print("\nSelf-attention comparison (first 5 positions):")
        print(f"Original:\n{orig_self_attn[:, :5]}")
        print(f"Modified:\n{mod_self_attn[:, :5]}")
        print(f"Max difference: {(orig_self_attn[:, :5] - mod_self_attn[:, :5]).abs().max()}")

        print("\nCross-attention comparison (first 5 positions):")
        print(f"Original:\n{orig_cross_attn[:, :5]}")
        print(f"Modified:\n{mod_cross_attn[:, :5]}")
        print(f"Max difference: {(orig_cross_attn[:, :5] - mod_cross_attn[:, :5]).abs().max()}")

        print("\nFinal output comparison (first 5 positions):")
        print(f"Original:\n{output1[:, :5]}")
        print(f"Modified:\n{output2[:, :5]}")
        print(f"Max difference: {(output1[:, :5] - output2[:, :5]).abs().max().item()}")

        # Check components separately
        self.assertTrue(
            torch.allclose(orig_self_attn[:, :5], mod_self_attn[:, :5], atol=1e-6),
            "Self attention leaks information",
        )

        self.assertTrue(
            torch.allclose(orig_cross_attn[:, :5], mod_cross_attn[:, :5], atol=1e-6),
            "Cross attention leaks information",
        )

        self.assertTrue(
            torch.allclose(output1[:, :5], output2[:, :5], atol=1e-6), "Decoder layer leaks information"
        )

    def test_attention_masking(self):
        """Test attention masking at a more granular level."""
        batch_size = 2
        src_len = 10
        tgt_len = 5

        # Create sample data
        src = torch.randn(batch_size, src_len, self.config["input_dim"]).to(self.device)
        tgt = torch.zeros(batch_size, tgt_len + 1, self.config["input_dim"]).to(self.device)
        tgt[:, 0, :] = src[:, -1, :]
        seq_values = torch.arange(1, tgt_len + 1).float().unsqueeze(-1)
        tgt[:, 1:, :] = seq_values.repeat(1, self.config["input_dim"])

        # Get mask
        tgt_mask = self.model._generate_square_subsequent_mask(tgt_len + 1).to(self.device)

        # Test single decoder layer
        decoder_layer = self.model.decoder_layers[0]

        # Register hook to get attention weights
        attention_weights = []

        def attention_hook(module, input, output):
            # Get attention weights from output
            attention_weights.append(output[1].detach().cpu())  # [batch_size, num_heads, tgt_len, tgt_len]

        hook = decoder_layer.self_attn.register_forward_hook(attention_hook)

        # Forward pass
        with torch.no_grad():
            decoder_output = decoder_layer(tgt, encoder_output=src, tgt_mask=tgt_mask)

        hook.remove()

        # Get attention weights for first head
        attn_weights = attention_weights[0][0, 0]  # [tgt_len, tgt_len]

        print("\nAttention weights for position 1:")
        print(attn_weights[1])  # Should only attend to position 0

        # Position 1 should only attend to position 0
        self.assertTrue(
            torch.allclose(attn_weights[1, 2:], torch.zeros_like(attn_weights[1, 2:])),
            "Position 1 is attending to future positions",
        )

        # Check if position 1 mainly attends to position 0
        self.assertTrue(attn_weights[1, 0] > 0.5, "Position 1 should primarily attend to position 0")
