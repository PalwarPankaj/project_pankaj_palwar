# --- train.py ---
import torch
import os
from tqdm import tqdm
import config

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_model(model, num_epochs, train_loader, loss_fn, optimizer, 
                val_loader=None, device=None):
    """
    Runs the full training loop for num_epochs.
    Saves best model weights to checkpoints/final_weights.pth
    """
    if device is None:
        device = get_device()

    model.to(device)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        loop = tqdm(train_loader, leave=False,
                    desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for inputs, labels in loop:          # ← fixed: only 2 values
            inputs  = inputs.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss        += loss.item()
            _, predicted         = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions   += labels.size(0)

            loop.set_postfix(
                loss=running_loss / (loop.n + 1),
                acc=f"{100 * correct_predictions / total_predictions:.1f}%"
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc  = 100 * correct_predictions / total_predictions
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # --- VALIDATION PHASE ---
        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, 
                                               loss_fn, device)
            print(f"              "
                  f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           "checkpoints/final_weights.pth")
                print(f"              ✅ Best model saved "
                      f"(val_acc={val_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")
    return model


def evaluate(model, dataloader, loss_fn, device=None):
    """
    Evaluates model on a dataloader.
    Returns: (epoch_loss, accuracy, all_preds, all_labels)
    """
    if device is None:
        device = get_device()

    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for inputs, labels in dataloader:    # ← fixed: only 2 values
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    acc        = 100 * correct / total

    return epoch_loss, acc, all_preds, all_labels