# --- dataset.py ---

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class CervicalCellDataset(Dataset):
    def __init__(self, dataset_path=config.dataset_path, is_train=False):
        self.dataset_path = dataset_path

        # Get class folders (Abnormal, Benign, Normal) — sorted for consistency
        self.classes = sorted([
            folder for folder in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, folder))
        ])

        self.class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}

        # Transform used during validation / inference
        self.test_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Transform used during training (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform = self.train_transform if is_train else self.test_transform

        # Build sample list: (full_image_path, integer_label)
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(dataset_path, cls)
            for filename in os.listdir(cls_path):
                if filename.lower().endswith(".bmp"):
                    full_path = os.path.join(cls_path, filename)
                    label = self.class_to_label[cls]
                    self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")   # ensure 3 channels
        image = self.transform(image)
        return image, label                            # ← always exactly 2 values

    def stratified_split(self, train_ratio=0.8, seed=42):
        """
        Splits dataset by class so each class is proportionally represented
        in both train and test sets.
        """
        random.seed(seed)

        class_to_samples = {cls: [] for cls in self.classes}
        for img_path, label in self.samples:
            cls = self.classes[label]
            class_to_samples[cls].append((img_path, label))

        train_samples = []
        test_samples  = []

        for cls, samples in class_to_samples.items():
            random.shuffle(samples)
            split_idx = int(train_ratio * len(samples))
            train_samples.extend(samples[:split_idx])
            test_samples.extend(samples[split_idx:])

        train_dataset         = CervicalCellDataset(self.dataset_path, is_train=True)
        test_dataset          = CervicalCellDataset(self.dataset_path, is_train=False)
        train_dataset.samples = train_samples
        test_dataset.samples  = test_samples

        return train_dataset, test_dataset


def get_dataloader(dataset, batch_size=config.batch_size, shuffle=config.shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)