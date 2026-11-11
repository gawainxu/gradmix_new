import torch
import torch.nn as nn
import torch.nn.functional as F


class basic_CNN(nn.Module):

    def __init__(self, embedding_dim=32, in_channels=2, num_classes=3):
        super().__init__()

        self.features = (nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        ))

        self.fc1 = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x, normalize=False):
        """x: [batch_size, 2, signal_length]."""
        embedding = F.relu(self.fc1(self.features(x)))

        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward(self, x, return_embedding=False):
        embedding = self.encode(x)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits
