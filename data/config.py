import torch

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATA / IMAGE PARAMS
# =========================
image_size = 320


# =========================
# MODEL PARAMS
# =========================
encoder = 'timm-efficientnet-b0'
weights = "imagenet"       

# =========================
# TRAINING PARAMS
# =========================
LR = 0.003                 # learning rate
epochs = 10                 # number of epochs
batch_size = 16
epochs = 25

