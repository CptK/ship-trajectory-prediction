import math

import torch
import torch.nn as nn


class TrajectoryEmbedding(nn.Module):
    """A PyTorch module that embeds sequential trajectory data with positional encoding.

    This module processes sequential input data by:
    1. Projecting the input features to a specified embedding dimension
    2. Adding positional encodings using sinusoidal functions
    3. Applying layer normalization and dropout

    The positional encoding follows the implementation from "Attention Is All You Need" (Vaswani et al.,
    2017), using fixed sinusoidal functions that allow the model to generalize to sequence lengths not seen
    during training.

    Attributes:
        input_projection (nn.Linear): Linear layer that projects input features to model dimension
        norm (nn.LayerNorm): Layer normalization applied after position encoding
        dropout (nn.Dropout): Dropout layer for regularization
        pe (torch.Tensor): Buffer containing precomputed positional encodings

    Example:
        >>> embedding = TrajectoryEmbedding(input_dim=6, d_model=128)
        >>> x = torch.randn(32, 100, 6)  # batch_size=32, seq_len=100, features=6
        >>> mask = torch.ones(32, 100, dtype=torch.bool)  # optional mask
        >>> output = embedding(x, mask)  # shape: (32, 100, 128)
    """

    def __init__(self, input_dim: int, d_model: int, max_seq_len: int = 1000, dropout: float = 0.1):
        """Initialize the TrajectoryEmbedding module.

        Args:
            input_dim: Dimension of input features.
            d_model: Dimension of the model's embeddings. This is the size that the input features will be
                projected to. Should match the model's expected input dimension if using with transformers.
            max_seq_len: Maximum sequence length this embedding will support. Default is 1000. Sequences
                longer than this will raise an error. Can be set higher but increases memory usage.
            dropout: Dropout probability for regularization. Default is 0.1.

        Raises:
            ValueError: If input_dim or d_model are not positive integers
            ValueError: If d_model is not even (required for positional encoding)
            ValueError: If max_seq_len is not positive
            ValueError: If dropout is not in range [0, 1]
        """
        super().__init__()

        # Input validation
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for positional encoding")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        # Project input features to hidden dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Create fixed positional encodings
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Optional mask for padded positions
        """
        # Project features
        x = self.input_projection(x)

        # apply layer normalization before adding positional encoding to better condition the input
        x = self.norm(x)

        # Add positional encoding
        seq_len = x.size(1)
        pos_encoding = self.pe[:seq_len]

        # If we have a mask, zero out positional encoding for padded positions
        if mask is not None:
            pos_encoding = pos_encoding.unsqueeze(0) * (~mask).float().unsqueeze(-1)

        # add the positional encoding to the input features
        x = x + pos_encoding.unsqueeze(0)  # broadcast across batch

        return self.dropout(x)  # type: ignore


class TrajectoryEncoder(nn.Module):
    """A PyTorch module that encodes trajectory sequences using a transformer encoder.

    This module processes embedded trajectory sequences using a stack of transformer  encoder layers.
    Each layer contains:

    1. Multi-head self-attention mechanism
    2. Position-wise feedforward network
    3. Layer normalization and residual connections
    4. Dropout for regularization

    The encoder can handle:
    - Variable length sequences through padding masks
    - Causal/autoregressive attention through source masks
    - Parallel processing of entire sequences

    Attributes:
        encoder (nn.TransformerEncoder): Stack of transformer encoder layers

    Example:
        >>> encoder = TrajectoryEncoder(
        ...     d_model=128,
        ...     nhead=8,
        ...     num_layers=6,
        ...     dim_feedforward=512
        ... )
        >>> x = torch.randn(32, 100, 128)  # batch_size=32, seq_len=100, d_model=128
        >>> mask = torch.ones(32, 100, dtype=torch.bool)  # optional padding mask
        >>> output = encoder(x, src_key_padding_mask=mask)  # shape: (32, 100, 128)
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        """Initialize the TrajectoryEncoder module.

        Args:
            d_model: Dimension of the model's hidden states. Must match the embedding dimension from
                TrajectoryEmbedding. Should be divisible by nhead.
            nhead: Number of attention heads. Each head will have dimension d_model/nhead. More heads allow
                the model to attend to different aspects of the sequence.
            num_layers: Number of transformer encoder layers to stack. Deeper networks can model more complex
                patterns but are harder to train.
            dim_feedforward: Dimension of the feedforward network within each encoder layer.
            dropout: Dropout probability for regularization. Default is 0.1.

        Raises:
            ValueError: If d_model is not divisible by nhead
            ValueError: If any dimension parameters are not positive
            ValueError: If dropout is not in range [0, 1]
        """
        super().__init__()

        # Input validation
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(nhead, int) or nhead <= 0:
            raise ValueError("nhead must be a positive integer")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(dim_feedforward, int) or dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be a positive integer")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # input shape: (batch_size, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a sequence of embeddings through the transformer encoder.

        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model) containing embedded sequence data.
            src_mask: Optional attention mask of shape (seq_len, seq_len) used to mask future positions in the
                sequence (for causal attention) or to prevent attention to certain positions.
            src_key_padding_mask: Optional boolean mask of shape (batch_size, seq_len) where True indicates
                positions that should be masked due to padding. This is typically used for variable length
                sequences within a batch.

        Returns:
            torch.Tensor: Encoded sequence tensor of shape (batch_size, seq_len, d_model) where each position
                contains contextual information from the entire sequence (subject to masking).
        """
        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # type: ignore


class TrajectoryDecoder(nn.Module):
    """A transformer-based decoder for processing trajectory sequences.

    This module implements a stack of transformer decoder layers that process target sequences while attending
    to encoded memory from the encoder. It is particularly useful for tasks like trajectory prediction, where
    future positions need to be predicted based on past observations.

    The decoder uses multi-head self-attention to process the target sequence and cross-attention to attend to
    the encoder's memory. Each decoder layer consists of:
    1. Self-attention on the target sequence
    2. Cross-attention between target and memory
    3. Feedforward neural network

    Each sub-layer is followed by dropout and layer normalization.

    Attributes:
        decoder (nn.TransformerDecoder): The main decoder stack containing multiple
            transformer decoder layers.

    Example:
        >>> decoder = TrajectoryDecoder(
        ...     d_model=256,
        ...     nhead=8,
        ...     num_layers=6,
        ...     dim_feedforward=1024
        ... )
        >>> tgt = torch.randn(32, 10, 256)  # target sequence
        >>> memory = torch.randn(32, 20, 256)  # encoder output
        >>> output = decoder(tgt, memory)  # shape: (32, 10, 256)
    """

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1
    ) -> None:
        """Initialize the TrajectoryDecoder.

        Args:
            d_model: Dimension of the model's embeddings. Must match the encoder's dimension. Should be
                divisible by nhead.
            nhead: Number of attention heads. The d_model will be split into nhead pieces for multi-head
                attention. Must divide d_model evenly.
            num_layers: Number of decoder layers in the stack.
            dim_feedforward: Dimension of the feedforward network model in each  decoder layer.
            dropout: Dropout probability applied after each sub-layer. Default is 0.1.

        Raises:
            ValueError: If d_model is not divisible by nhead
            ValueError: If any dimension is not positive
            ValueError: If dropout is not in range [0, 1]
        """
        super().__init__()

        # Input validation
        if not all(isinstance(x, int) and x > 0 for x in [d_model, nhead, num_layers, dim_feedforward]):
            raise ValueError("All dimension arguments must be positive integers")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        # Create a single decoder layer with specified parameters
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Use (batch, seq, feature) ordering
        )

        # Create the full decoder by stacking multiple layers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process the target sequence while attending to encoded memory.

        Args:
            tgt: Target sequence tensor of shape (batch_size, tgt_len, d_model). In trajectory prediction,
                this could be the sequence to be predicted.
            memory: Memory tensor from encoder of shape (batch_size, src_len, d_model). Contains the encoded
                representation of the input sequence.
            tgt_mask: Attention mask for target sequence of shape (tgt_len, tgt_len). Usually a causal mask to
                prevent attending to future positions.
            memory_mask: Attention mask for memory of shape (tgt_len, src_len). Usually None, as decoder can
                attend to all memory positions.
            tgt_key_padding_mask: Mask for padded positions in target sequence of shape (batch_size, tgt_len).
                True indicates position should be ignored.
            memory_key_padding_mask: Mask for padded positions in memory of shape (batch_size, src_len).
                True indicates position should be ignored.

        Returns:
            torch.Tensor: Decoded sequence of shape (batch_size, tgt_len, d_model)

        Raises:
            RuntimeError: If input tensor dimensions don't match expected shapes
            RuntimeError: If d_model of inputs doesn't match decoder's d_model
        """
        return self.decoder(  # type: ignore
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class TrajectoryTransformer(nn.Module):
    """Transformer model for trajectory prediction and denoising.

    This model uses a transformer architecture to process trajectory sequences, with support for various
    trajectory features like speed, orientation, and timestamps. It employs an encoder-decoder architecture
    with self-attention mechanisms to denoise and predict trajectory patterns.

    Attributes:
        input_dim (int): Dimension of input features
        embedding (TrajectoryEmbedding): Embedding layer for input features
        encoder (TrajectoryEncoder): Transformer encoder
        decoder (TrajectoryDecoder): Transformer decoder
        output_projection (nn.Linear): Projects transformer output to feature space
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        include_speed: bool = True,
        include_orientation: bool = True,
        include_timestamps: bool = True,
    ):
        """Initialize the TrajectoryTransformer.

        Args:
            d_model: Dimension of the transformer model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            include_speed: Whether to include speed features
            include_orientation: Whether to include orientation features
            include_timestamps: Whether to include timestamp features
        """
        super().__init__()

        # Calculate input dimension based on included features
        self.input_dim = 2  # base features: lat, lon
        if include_speed:
            self.input_dim += 1  # speed magnitude
        if include_orientation:
            self.input_dim += 2  # sin and cos of heading
        if include_timestamps:
            self.input_dim += 1  # temporal information

        # Model components
        self.embedding = TrajectoryEmbedding(self.input_dim, d_model, max_seq_len, dropout)
        self.encoder = TrajectoryEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = TrajectoryDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)

        # Replace TrajectoryHead with direct linear projection
        self.output_projection = nn.Linear(d_model, self.input_dim)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Process input trajectory sequence.

        Args:
            src: Input sequence tensor of shape (batch_size, seq_len, input_dim)
                containing trajectory features:
                - Positions (lat, lon) [required]
                - Speed [optional]
                - Orientation (sin, cos) [optional]
                - Timestamps [optional]
            src_key_padding_mask: Boolean mask for padded positions in source sequence of shape
                (batch_size, seq_len). True indicates position should be masked.

        Returns:
            torch.Tensor: Predicted/denoised trajectory of shape (batch_size, seq_len, input_dim)
        """
        # Embed input sequence
        src_embedded = self.embedding(src)

        # Encode sequence
        memory = self.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)

        # Decode sequence
        decoded = self.decoder(
            src_embedded,  # Use same masked input for denoising
            memory,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Project directly to output space
        return self.output_projection(decoded)  # type: ignore
