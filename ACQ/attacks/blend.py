from torchvision import transforms
from basic_attack import basic_attacker
import copy
import numpy as np
import torch


class Blend(basic_attacker):
    def __init__(self, config: dict, test: bool) -> None:
        super().__init__(config, test)
        self.name = 'blend'
        self.alpha = 0.2

    def make_trigger(self, sample: np.ndarray) -> np.ndarray:
        loader = transforms.ToTensor()
        unloader = transforms.ToPILImage()

        data = copy.deepcopy(sample)
        data = loader(data)

        mask = torch.randn(data.shape)
        blend_data = (1-self.alpha)*data + self.alpha*mask
        blend_data.clamp_(0., 1.)
        blend_data = unloader(blend_data)

        return blend_data
