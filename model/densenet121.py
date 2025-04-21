import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from model.attention import SELayer, CBAMLayer
from config import config

class PretrainedDenseNet121(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(PretrainedDenseNet121, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Replacing the classifier
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, config.num_classes)
        )

        # Freezing all the layers
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreezing only the classification layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class SE_PretrainedDenseNet121(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(SE_PretrainedDenseNet121, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # adding SE layers after each denseblock
        self.model.features.denseblock1 = nn.Sequential(self.model.features.denseblock1, SELayer(256))
        self.model.features.denseblock2 = nn.Sequential(self.model.features.denseblock2, SELayer(512))
        self.model.features.denseblock3 = nn.Sequential(self.model.features.denseblock3, SELayer(1024))

        # Replacing the classifier
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, config.num_classes)
        )

        # Freezing all the layers
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreezing SE layers
        for name, module in self.model.features.named_modules():
            if isinstance(module, SELayer):
                for param in module.parameters():
                    param.requires_grad = True

        # unfreezing only the classification layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)