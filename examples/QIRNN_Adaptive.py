import torch
import torch.nn as nn
from qirnn import QIRNN_Adaptive

# Define input parameters
vocab_size = 10000
embedding_dim = 128
hidden_dim = 256
num_layers = 3
output_dim = 10
heads = 4
batch_size = 32
seq_length = 20

# Create random input sequence
x = torch.randint(0, vocab_size, (batch_size, seq_length))

# Initialize QIRNN_Adaptive model
model_adaptive = QIRNN_Adaptive(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, heads)

# Forward pass
output_adaptive = model_adaptive(x)

# Print output shape
print("Output Shape:", output_adaptive.shape)
