# --- config.py ---

import os

# Path to the 3-class dataset folder.
# Use an environment variable if set, otherwise fall back to a relative path.
dataset_path = os.environ.get("DATASET_PATH", "SIPaKMeD_3_Classes")

classes        = ["Abnormal", "Benign", "Normal"]
num_classes    = len(classes)

# Image dimensions (must match model input)
image_size     = (128, 128)
resize_x       = 128
resize_y       = 128
input_channels = 3

# Training hyperparameters
batch_size     = 32
shuffle        = True
epochs         = 70
learning_rate  = 0.001

weight_decay   = 1e-4       # L2 regularisation

# Checkpoint
checkpoint_path = "checkpoints/final_weights.pth"   # ← exact name professor requires

log = True

# Custom CNN architecture parameters
conv1_out_channels = 32
conv2_out_channels = 64
conv3_out_channels = 128
conv4_out_channels = 256
fc1_out_features   = 512
fc2_out_features   = 128

training_split = {
    "train": 0.8,
    "test":  0.2
}