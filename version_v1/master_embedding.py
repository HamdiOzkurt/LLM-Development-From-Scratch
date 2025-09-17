import torch
import torch.nn as nn


def get_rotary_position_encoding(input: torch.Tensor, base=10000, device="cpu"):
    """
    Apply rotary position encoding to input tensor.
    
    Args:
        input: Tensor of shape (batch_size, sequence_length, embedding_dim) or (sequence_length, embedding_dim)
        base: Base for frequency calculation
        device: Device to use for computations
    
    Returns:
        Tensor with rotary position encoding applied
    """
    # Handle both 2D and 3D tensors
    if len(input.shape) == 3:
        batch_size, context_length, dimension = input.shape
        # Reshape to 2D for processing
        input_2d = input.view(-1, dimension)  # (batch_size * context_length, dimension)
        
        # Create position indices for each sequence in the batch
        positions = torch.arange(0, context_length, device=device, dtype=torch.float32)
        positions = positions.repeat(batch_size).unsqueeze(1)  # (batch_size * context_length, 1)
        
    elif len(input.shape) == 2:
        context_length, dimension = input.shape
        input_2d = input
        positions = torch.arange(0, context_length, device=device, dtype=torch.float32).unsqueeze(1)
        batch_size = 1
    else:
        raise ValueError(f"Input tensor must be 2D or 3D, got shape {input.shape}")

    assert dimension % 2 == 0, "dimension must be even"

    half_dimension = dimension // 2

    # Calculate frequency indices
    freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (freqs_indices / dimension))

    # Calculate angles
    angles = positions * freqs

    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    # Split input into even and odd components
    input_even = input_2d[:, :dimension // 2]  # [0, 2, 4, ..]
    input_odd = input_2d[:, dimension // 2:]   # [1, 3, 5, ..]

    # Apply rotation
    input_even_rotated = input_even * cos_angles - input_odd * sin_angles
    input_odd_rotated = input_even * sin_angles + input_odd * cos_angles
    
    # Reconstruct the rotated input
    input_rotated = torch.empty_like(input_2d)
    input_rotated[:, :dimension // 2] = input_even_rotated
    input_rotated[:, dimension // 2:] = input_odd_rotated

    # Reshape back to original shape if needed
    if len(input.shape) == 3:
        input_rotated = input_rotated.view(batch_size, context_length, dimension)

    return input_rotated


class MasterEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Store parameters for position encoding
        self.embedding_dim = embedding_dim
        self.context_length = context_length

    def forward(self, x):
        """
        Forward pass of the embedding layer.
        
        Args:
            x: Input token indices, shape (batch_size, sequence_length) or (sequence_length,)
        
        Returns:
            Embedded tokens with rotary position encoding applied
        """
        # Get token embeddings
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        
        # Apply rotary position encoding
        device = x.device
        x = get_rotary_position_encoding(x, device=device)
        
        return x


# Alternative version without rotary encoding (simpler fallback)
class MasterEmbeddingSimple(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Traditional position embedding as fallback
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, x):
        """
        Forward pass with traditional position encoding.
        """
        seq_len = x.size(-1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        token_emb = self.embedding(x)
        pos_emb = self.pos_embedding(pos_ids)
        
        return token_emb + pos_emb
