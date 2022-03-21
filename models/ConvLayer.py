from torch import nn
from torch.nn import functional as F

class ConvLayer(nn.Module):
    def __init__(self, config, is_multires= False) -> None:
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=config['cnn1_in'], out_channels=config['cnn2_in'], kernel_size=11, padding=1, stride=4)
        self.batchnorm1 = nn.BatchNorm2d(config['cnn2_in'])
        self.conv2 = nn.Conv2d(in_channels=config['cnn2_in'], out_channels=config['cnn3_in'], kernel_size=5, padding=1, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(config['cnn3_in'])
        self.conv3 = nn.Conv2d(in_channels=config['cnn3_in'], out_channels=config['cnn4_in'], kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=config['cnn4_in'], out_channels=config['cnn5_in'], kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=config['cnn5_in'], out_channels=config['cnn5_out'], kernel_size=3, padding=1, stride=1)

        self.pooling = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(config['dropout'])
        self.is_multires = is_multires

    def forward(self, x):           # N x 3 x 170 x 170
        x = self.conv1(x)           # N x 32
        x = self.batchnorm1(x)
        x = self.pooling(x)
        x = F.leaky_relu(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.pooling(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.dropout(x)

        x = self.conv4(x)
        x = F.leaky_relu(x)

        x = self.dropout(x)

        x = self.conv5(x)
        if not self.is_multires:
            x = self.pooling(x)
        x = F.leaky_relu(x)

        return x                    # N x 128 x 2 x 2