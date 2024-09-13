import torch.nn as nn
import torch


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, hub_path, num_classes=1):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2',
                                    hub_path)
        embeddings_size = self.model.norm.weight.shape[0]
        # Todo - check making classifier with more params compare to RadImageNet
        self.classifier = nn.Sequential(
            nn.Linear(embeddings_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.model.norm(x)
        x = self.classifier(x)
        return x
