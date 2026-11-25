import torch
import torch.nn as nn
import torch.nn.functional as F


class basic_CNN(nn.Module):

    def __init__(self, feature_dim=32, in_channels=2, num_classes=3):
        super().__init__()

        self.features = (nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        ))

        self.fc1 = nn.Linear(64, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def encode(self, x, normalize=False):
        """x: [batch_size, 2, signal_length]."""
        embedding = F.relu(self.fc1(self.features(x)))

        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward(self, x, return_embedding=True):
        embedding = self.encode(x)
        logits = self.classifier(embedding)

        if return_embedding:
            return embedding
        
        return logits


class basic_CNN_multihead(nn.Module):

    def __init__(self, feature_dim=32, in_channels=2, num_classes=3):
        super().__init__()

        self.features = (nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        ))

        self.fc11 = nn.Linear(64, feature_dim)
        self.fc12 = nn.Linear(64, feature_dim)
        self.fc13 = nn.Linear(64, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def encode(self, x, normalize=False):
        """x: [batch_size, 2, signal_length]."""
        embedding1 = F.relu(self.fc11(self.features(x)))
        embedding2 = F.relu(self.fc12(self.features(x)))
        embedding3 = F.relu(self.fc13(self.features(x)))

        if normalize:
            embedding1 = F.normalize(embedding1, p=2, dim=1)
            embedding2 = F.normalize(embedding2, p=2, dim=1)
            embedding3 = F.normalize(embedding3, p=2, dim=1)

        return embedding1, embedding2, embedding3

    def forward(self, x):
        embedding1, embedding2, embedding3 = self.encode(x)

        return embedding1, embedding2, embedding3


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=10, feat_dim=32):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, features):
        return self.fc(features)


if __name__ == '__main__':

    model = basic_CNN()
    # Print all layer names and their tensor shapes/parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Num Elements: {param.numel()}")