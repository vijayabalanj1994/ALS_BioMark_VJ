import torch.nn as nn

class MultiCNNModel(nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(MultiCNNModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.flatten = nn.Flatten()

        self.cnn= nn.Sequential(
            # block 1
            nn.Conv2d(3,32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate),

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate),

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate),
        )

        # placeholders for fully connect layer, will be defined dynamically in forward method
        self.fc_input_size =None
        self.fc = None

    def forward(self, x):

        x = self.cnn(x)

        if not self.fc_input_size:
            self.fc_input_size = x.size(1)* x.size(2) * x.size(3)
            self.fc = nn.Sequential(
                nn.Linear(self.fc_input_size, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, 50),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(50, 3)
            ).to(x.device)

        x = self.flatten(x)
        x = self.fc(x)
        return x