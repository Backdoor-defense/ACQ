from torch.utils.data import Dataset
import inspect
import torch.nn as nn
import torch
from torch import Tensor
import numpy as np
from torchvision import transforms
from typing import Dict, Iterable, Callable
import os


def makedir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        pass


def cfunc(func):
    lc = locals()
    func_code = inspect.getsource(func)
    func_code = func_code.replace('logger.make_checkpoints(mode)', 'pass')
    print(func_code)
    exec(func_code)
    res_func = lc[func.__name__]
    return res_func


class splited_dataset(Dataset):
    def __init__(self, label: int, transforms: transforms.Compose) -> None:
        super().__init__()
        self.data = []
        self.label = label
        self.transforms = transforms

    def make_labels(self):
        self.labels = [self.label]*len(self.data)

    def add_data(self, data_: np.ndarray):
        self.data.append(data_)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if index <= self.__len__() - 1:
            image = self.data[index]
            label = self.labels[index]
            return self.transforms(image), label
        else:
            raise ValueError('Too large index for splited dataset!')


def factorization(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = int(num / k)
                break
    if len(factor) == 1:
        return [1, factor[0]]

    return factor
