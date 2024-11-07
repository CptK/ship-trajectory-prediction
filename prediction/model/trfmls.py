import torch
import torch.nn as nn


class TRFMLSEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        max_seq_len: int,
        dropout: float = 0.1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        # Embedding and positional encoding for this layer
        self.input_embedding = nn.Linear(d_model, d_model)  # Maintains dimension
        self.pos_encoder = nn.Sequential(nn.Embedding(max_seq_len, d_model), nn.Dropout(dropout))

        # LSTM for temporal feature processing
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=batch_first)

        # Transformer encoder layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Fusion layer
        self.fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Apply positional encoding and embedding for this layer
        src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        src = self.input_embedding(src) + self.pos_encoder[0](src_pos)

        # Process with LSTM
        lstm_out, _ = self.lstm(src)

        # Process with transformer encoder layer
        transformer_out = self.transformer_encoder_layer(src, src_mask)

        # Fuse LSTM and transformer outputs
        combined = torch.cat([lstm_out, transformer_out], dim=-1)
        fused_output = self.fusion(combined)

        return fused_output


class TRFMLS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        output_dim: int,
        dropout: float = 0.1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        # Initial input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TRFMLSEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, output_dim),
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Initial projection
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)

        # Generate causal mask for decoder
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Maybe we also need a cross-attention mask?
        memory_mask = torch.zeros(tgt.size(1), src.size(1)).to(tgt.device)

        # Pass through encoder layers
        encoder_output = src
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)

        # Pass through decoder layers
        decoder_output = tgt
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output,
                encoder_output,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,  # Add cross-attention mask
            )

        output = self.output_layer(decoder_output)
        return output

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float("-inf")).masked_fill(~mask, 0.0)
        return mask


if __name__ == "__main__":
    # Example configuration
    config = {
        "input_dim": 4,  # Input features
        "d_model": 256,  # Model dimension
        "nhead": 8,  # Number of attention heads
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 1024,
        "max_seq_len": 100,
        "output_dim": 2,  # Output features
        "dropout": 0.1,
        "batch_first": True,
    }

    # Create model
    model = TRFMLS(**config)

    # Example input tensors
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 30

    src = torch.randn(batch_size, src_seq_len, config["input_dim"])
    tgt = torch.randn(batch_size, tgt_seq_len, config["input_dim"])

    # Forward pass
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
