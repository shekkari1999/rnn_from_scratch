# config.py
import torch

# General settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset settings
MAX_LENGTH = 128
BATCH_SIZE = 1
MIN_FREQ = 3

# Model settings
EMBEDDING_DIM = 50  # Glove embedding size
HIDDEN_DIM = 100  # RNN hidden size
NUM_EPOCHS = 1
LEARNING_RATE = 0.001