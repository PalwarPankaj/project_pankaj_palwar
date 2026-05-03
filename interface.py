# --- interface.py ---
# This file standardises all names for the grading program.
# DO NOT rename your functions — only update the imports here.

# The grading program will use TheModel, the_trainer, the_predictor,
# TheDataset, the_dataloader, the_batch_size, and total_epochs.

from model import CustomCervicalCNN as TheModel

from train import train_model as the_trainer

from predict import classify_cells as the_predictor

from dataset import CervicalCellDataset as TheDataset

from dataset import get_dataloader as the_dataloader

from config import batch_size as the_batch_size

from config import epochs as total_epochs