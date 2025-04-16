import torch.nn as nn
import torchvision.models as models
from config import config

class PretrainedResNet18(nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(PretrainedResNet18, self).__init__()

        self.dropout_rate = dropout_rate

        self.resnet = models.resnet18(pretrained=True)

        # replacing the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_ftrs, config.num_classes)
        )

        # freezing all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreeze only the final layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)