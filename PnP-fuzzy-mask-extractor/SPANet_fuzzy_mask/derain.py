import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out

class derain(nn.Module):
    def __init__(self):
        super(derain, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(3, 32),
            nn.ReLU(True)
        )
        self.res_block1 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(32, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.conv_out = nn.Sequential(
            conv3x3(32, 3)
        )

    def forward(self, O, mask):
        
        out = self.conv_in(O)  # 256*256
        mask = self.conv_in(mask)
        # mask.detach_()
        # 经过res_block后图像大小不变
        out = F.relu(self.res_block1(out) + out)  # 256*256
        out = F.relu(self.res_block2(out) + out)  # 256*256
        out = F.relu(self.res_block3(out) + out)  # 256*256

        out = F.relu(self.res_block4(out) * mask + out)
        out = F.relu(self.res_block5(out) * mask + out)
        out = F.relu(self.res_block6(out) * mask + out)

        out = F.relu(self.res_block7(out) * mask + out)
        out = F.relu(self.res_block8(out) * mask + out)
        out = F.relu(self.res_block9(out) * mask + out)

        out = F.relu(self.res_block10(out) * mask + out)
        out = F.relu(self.res_block11(out) * mask + out)
        out = F.relu(self.res_block12(out) * mask + out)

        out = F.relu(self.res_block13(out) * mask + out)
        out = F.relu(self.res_block14(out) * mask + out)
        out = F.relu(self.res_block15(out) * mask + out)

        out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)

        out = self.conv_out(out)

        return out