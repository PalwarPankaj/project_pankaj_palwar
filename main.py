# --- main.py ---
# This is your master run script. Run this to train from scratch.
# It is NOT graded directly — the grader uses interface.py —
# but a working main.py shows the professor everything runs end-to-end.

import torch
from torch import optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config
from model import CustomCervicalCNN
from train import train_model, evaluate, get_device
from dataset import CervicalCellDataset, get_dataloader


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs,   label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved training_curves.png")
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted by CNN")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix on Validation Data")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")
    plt.show()


def main():
    device = get_device()
    print(f"Running on device: {device}")

    # ── 1. Load & split dataset ──────────────────────────────────────────
    full_dataset = CervicalCellDataset(is_train=False)
    train_dataset, val_dataset = full_dataset.stratified_split(
        train_ratio=config.training_split["train"]
    )

    train_loader = get_dataloader(train_dataset,
                                  batch_size=config.batch_size, shuffle=True)
    val_loader   = get_dataloader(val_dataset,
                                  batch_size=config.batch_size, shuffle=False)

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes           : {full_dataset.classes}")

    # ── 2. Build model, loss, optimiser ─────────────────────────────────
    model     = CustomCervicalCNN(num_classes=config.num_classes).to(device)
    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    os.makedirs("checkpoints", exist_ok=True)

    # ── 3. Train — single call, train_model handles the epoch loop ───────
    # Signature: train_model(model, num_epochs, train_loader, loss_fn,
    #                        optimizer, val_loader=None, device=None)
    trained_model = train_model(
        model        = model,
        num_epochs   = config.epochs,
        train_loader = train_loader,
        loss_fn      = loss_fn,
        optimizer    = optimizer,
        val_loader   = val_loader,
        device       = device
    )

    # ── 4. Final evaluation on validation set ───────────────────────────
    val_loss, val_acc, all_preds, all_labels = evaluate(
        trained_model, val_loader, loss_fn, device
    )
    print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Validation Loss    : {val_loss:.4f}")

    # ── 5. Plots ─────────────────────────────────────────────────────────
    plot_confusion_matrix(all_labels, all_preds, full_dataset.classes)
    print("\nDone. Model saved to:", config.checkpoint_path)


if __name__ == "__main__":
    main()