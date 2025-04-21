import torch
import torch.nn as nn

class SELayer(nn.Module):

    def __init__(self, input_channel_dim, reduction=16):

        super(SELayer, self).__init__()

        # average pooling to get a single value per channel
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # fully connected layer to get channel attention scores
        self.fc = nn.Sequential(
            nn.Linear(input_channel_dim, input_channel_dim // reduction, bias=False), #squeeze
            nn.ReLU(inplace=True),
            nn.Linear(input_channel_dim // reduction, input_channel_dim, bias=False), #excitation
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):

    def __init__(self, input_channel_dim, reduction=16):

        super(ChannelAttention, self).__init__()

        # average pooling to get a single value per channel
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # average pooling to get a single value per channel
        # max pooling to get a single value per channel
        self.max_pool = nn.AdaptiveAvgPool2d((1, 1))

        #shared MLP(for average and max) conv layers to get attention scores
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(input_channel_dim, input_channel_dim//reduction, kernel_size=1, bias=False), #squeeze
            nn.ReLU(),
            nn.Conv2d(input_channel_dim//reduction, input_channel_dim, kernel_size=1,bias=False) #excitation
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        #adding both
        out = avg_out+max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # padding to maintain input size
        padding = (kernel_size -1)//2
        # Conv to fuse avg and max out channels
        self.conv = nn.Conv2d(2,1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # for each pixel
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # concatenate, so new dim ->2
        x_cat = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAMLayer(nn.Module):
    def __init__(self, input_channel_dim, reduction=16, spatial_kernal=7):

        super(CBAMLayer, self).__init__()
        self.channel_attention = ChannelAttention(input_channel_dim, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernal)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x