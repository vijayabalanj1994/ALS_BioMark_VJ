import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from model.attention import SELayer
from config import config

class PretrainedResNet18(nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(PretrainedResNet18, self).__init__()

        # loading pre_trained resnet
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # replacing the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, config.num_classes)
        )

        # freezing all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreezing the final fully connected layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)

class SE_PretrainedResNet18(nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(SE_PretrainedResNet18, self).__init__()

        # loading pre_trained resnet
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # adding SE blocks after each residual stage
        self.resnet.layer1 = nn.Sequential(self.resnet.layer1, SELayer(64))
        self.resnet.layer2 = nn.Sequential(self.resnet.layer2, SELayer(128))
        self.resnet.layer3 = nn.Sequential(self.resnet.layer3, SELayer(256))
        self.resnet.layer4 = nn.Sequential(self.resnet.layer4, SELayer(512))

        # replacing the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, config.num_classes)
        )

        # freezing all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # unfreezing all SE Layers
        for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for module in layer.modules():
                if isinstance(module, SELayer):
                    for param in module.parameters():
                        param.requires_grad = True

        # Unfreezing the final fully connected layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)