# --- predict.py ---

import os
import torch
from PIL import Image
import config
from model import CustomCervicalCNN
from train import get_device
from torchvision import transforms

# Inference transform (no augmentation)
_infer_transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

_classes = config.classes  # ["Abnormal", "Benign", "Normal"]


def _load_model(model_path, device):
    model = CustomCervicalCNN(num_classes=config.num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}, using untrained model.")
    model.eval()
    model.to(device)
    return model


def inferloader(list_of_image_paths):
    """
    Converts a list of image paths into a batched tensor ready for the model.
    """
    images = []
    for image_path in list_of_image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = _infer_transform(image)   # shape: (3, 128, 128)
        images.append(image_tensor)
    return torch.stack(images, dim=0)            # shape: (N, 3, 128, 128)


def classify_cells(list_of_image_paths, model_path=config.checkpoint_path):
    """
    Takes a list of image file paths and returns a list of predicted class labels.
    This is the function the grading program calls (imported as the_predictor).

    Args:
        list_of_image_paths (list): List of paths to .bmp image files.
        model_path (str): Path to the saved model weights.

    Returns:
        list: Predicted class label string for each image.
    """
    device = get_device()
    model = _load_model(model_path, device)

    # ── batch all images at once (as professor requires) ──────────────────
    image_batch = inferloader(list_of_image_paths).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        _, predicted_indices = torch.max(logits, 1)

    labels = [_classes[idx] for idx in predicted_indices.cpu().numpy()]

    for path, label in zip(list_of_image_paths, labels):
        print(f"  {path}  →  {label}")

    return labels


# Quick test — run directly to verify predict works
if __name__ == "__main__":
    test_paths = []
    data_dir = "data"
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.lower().endswith(".bmp"):
                test_paths.append(os.path.join(data_dir, f))
                if len(test_paths) == 3:
                    break

    if test_paths:
        results = classify_cells(test_paths)
        print("\nResults:", results)
    else:
        print("No .bmp images found in data/ — add sample images to test.")