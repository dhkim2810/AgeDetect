import torch
import torch.nn as nn

'''ResNet18/34/50/101/152 in Pytorch.'''

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpinalResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SpinalResNet, self).__init__()
        self.first_HL = 256

        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, self.first_HL)
        self.fc1_1 = nn.Linear(512+self.first_HL, self.first_HL)
        self.fc1_2 = nn.Linear(512+self.first_HL, self.first_HL)
        self.fc1_3 = nn.Linear(512+self.first_HL, self.first_HL)

        self.fc_layer = nn.Linear(self.first_HL*4,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out1 = self.maxpool(out)
        out2 = out1[:,:,0,0]
        out2 = out2.view(out2.size(0), -1)

        x1 = out1[:,:,0,0]
        x1 = self.relu(self.fc1(x1))
        x2 = torch.cat([out1[:,:,0,1], x1], dim=1)
        x2 = self.relu(self.fc1_1(x2))
        x3 = torch.cat([out1[:,:,1,0], x2], dim=1)
        x3 = self.relu(self.fc1_2(x3))
        x4 = torch.cat([out1[:,:,1,1], x3], dim=1)
        x4 = self.relu(self.fc1_3(x4))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        out = torch.cat([x,x4], dim=1)
        out = self.linear(out)
        return out


def SpinalResNet18(num_classes=10):
    return SpinalResNet(BasicBlock, [2,2,2,2], num_classes)

def SpinalResNet34(num_classes=10):
    return SpinalResNet(BasicBlock, [3,4,6,3], num_classes)