import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from main_config import *
import sys
sys.path.append('/home/jyl/distill-defense/')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(
            model_config['num_classes'], 100)
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(
            nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )

        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(3, affine=False)
        )
        self.smooth = transforms.GaussianBlur(3)

    def forward(self, z, labels):
        gen_input = torch.mul(self.label_emb(labels), z)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = self.smooth(img)
        return img
