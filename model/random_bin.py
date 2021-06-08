import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv(in_channels, out_channels, kerner_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kerner_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )

class Classifier(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Classifier, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.Softmax(dim=1))
    def forward(self, x):
        return self.body(x)


class RandomBin(nn.Module):
    def __init__(self, N, M):
        super(RandomBin, self).__init__()
        self.conv1 = Conv(3, 16, 3, 1, 1)
        self.conv2 = Conv(16, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv(16, 32, 3, 1, 1)
        self.conv4 = Conv(32, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv(32, 64, 3, 1, 1)
        self.conv6 = Conv(64, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = Conv(64, 128, 3, 1, 1)
        self.conv8 = Conv(128, 128, 3, 1, 1)
        self.conv9 = Conv(128, 128, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv10 = Conv(128, 128, 3, 1, 1)
        self.conv11 = Conv(128, 128, 3, 1, 1)
        self.conv12 = Conv(128, 128, 3, 1, 1)
        self.HP = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )

        self.classifiers = nn.ModuleList([Classifier(128,N) for _ in range(M)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool5(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.HP(x)
        x = x.view((x.size(0), -1))
        new_x = torch.Tensor([]).cuda()
        for classifier in self.classifiers:
            tmp = classifier(x)
            new_x = torch.cat((new_x, tmp), dim=1)
        return new_x


def random_bin(M=30, N=10):
    return RandomBin(N, M)