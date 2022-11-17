import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class reslstm(nn.Module):
    def __init__(self):
        super(reslstm, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        # self.conv_i = nn.Sequential(
        #     nn.Conv2d(32 + 32, 32, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        # self.conv_f = nn.Sequential(
        #     nn.Conv2d(32 + 32, 32, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        # self.conv_g = nn.Sequential(
        #     nn.Conv2d(32 + 32, 32, 3, 1, 1),
        #     nn.Tanh()
        # )
        # self.conv_o = nn.Sequential(
        #     nn.Conv2d(32 + 32, 32, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        
    def forward(self,O):
        device = 'cuda'
        batch_size, row, col = O.size(0), O.size(2), O.size(3)
        times_in_attention = 4
        mask = Variable(torch.ones(batch_size, 3, row, col)).to(device) / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        c = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        mask_list = []
        attention_map = []
        for i in range(times_in_attention):
            x = torch.cat((O, mask), 1)
            x = self.det_conv0(x)
            # resx = x
            x = F.relu(self.det_conv1(x) + x)
            # resx = x
            x = F.relu(self.det_conv2(x) + x)
            # resx = x
            x = F.relu(self.det_conv3(x) + x)
            # resx = x
            x = F.relu(self.det_conv4(x) + x)
            # resx = x
            x = F.relu(self.det_conv5(x) + x)
            # x = torch.cat((x, h), 1)
            # attention_map.append(x)
            # i = self.conv_i(x)
            # f = self.conv_f(x)
            # g = self.conv_g(x)
            # o = self.conv_o(x)
            # c = f * c + i * g
            # h = o * torch.tanh(c)
            mask = self.det_conv_mask(x)
            # print(mask.shape)
            mask_list.append(mask)

        return mask_list