from torchvision import transforms, datasets
from basic_attack import basic_attacker
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import PIL.Image


def trigger():
    mask = 1*np.ones((32, 32, 3), dtype=np.uint8)
    for i in range(32):
        mask[:, i, :] = round(20*math.sin(2*math.pi*i*6/32))
    np.save('/home/jyl/distill-defense/attacks/sig_mask.npy', mask)


class SIG(basic_attacker):
    def __init__(self, config: dict, test: bool) -> None:
        super().__init__(config, test)
        self.name = 'SIG'
        self.alpha = 0.2

    def _trigger(self):
        trigger()

    def make_trigger(self, sample: np.ndarray) -> np.ndarray:
        alpha = 0.2
        signal_mask = np.load('/home/jyl/distill-defense/attacks/sig_mask.npy')
        data = copy.deepcopy(sample)
        sig_img = (1-alpha)*data+alpha*signal_mask
        sig_img = sig_img.astype(np.uint8)
        np.clip(sig_img, 0, 255)
        return sig_img
