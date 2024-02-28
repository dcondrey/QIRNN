import torch
import torch.nn as nn
from qirnn import QuantumAdaptiveComputationTime

# Define input parameters
embedding_dim = 64
max_steps = 5
seq_length = 10
batch_size = 32

# Create random input quantum state
quantum_state = torch.randn(batch_size, seq_length, embedding_dim)

# Initialize QuantumAdaptiveComputationTime module
qact = QuantumAdaptiveComputationTime(embedding_dim, max_steps)

# Forward pass
pondered_state, remainders, n_updates = qact(quantum_state)

# Print output shapes
print("Pondered State Shape:", pondered_state.shape)
print("Remainders Shape:", remainders.shape)
print("Number of Updates Shape:", n_updates.shape)
