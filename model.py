# --- model.py ---
import torch
import torch.nn as nn
import config

class CustomCervicalCNN(nn.Module):
    def __init__(self, num_classes=config.num_classes):  # ← default added
        super(CustomCervicalCNN, self).__init__()

        # Block 1 (Input: 3 channels RGB)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, config.conv1_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(config.conv1_out_channels, config.conv2_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(config.conv2_out_channels, config.conv3_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.conv3_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(config.conv3_out_channels, config.conv4_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.conv4_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self._flattened_size = self._get_flattened_size()

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(self._flattened_size, config.fc1_out_features),
            nn.BatchNorm1d(config.fc1_out_features),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc1_out_features, config.fc2_out_features),
            nn.BatchNorm1d(config.fc2_out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Linear(config.fc2_out_features, num_classes)

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, *config.image_size)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # Raw logits (CrossEntropyLoss handles softmax)