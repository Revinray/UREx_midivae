import torch
import torch.nn as nn

# Define the embedding layer
vocab_size = 10000
embedding_dim = 128
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Define a simple encoder layer (e.g., a single Transformer encoder layer)
encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Example token indices
token_indices = torch.tensor([1, 2, 3, 4, 5])

# Embedding step
embedded_tokens = embedding_layer(token_indices)

# print(embedded_tokens)
# print its shape
print(embedded_tokens.shape)

# Encoding step
encoded_tokens = encoder(embedded_tokens.unsqueeze(1))

# print(encoded_tokens)