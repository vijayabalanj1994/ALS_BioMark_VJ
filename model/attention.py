import torch.nn as nn

class SELayer(nn.Module):

    def __init__(self, input_dim, reduction=16):

        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim//reduction, bias=False), #squeeze
            nn.ReLU(inplace=True),
            nn.Linear(input_dim//reduction, input_dim, bias=False), #excitation
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)
