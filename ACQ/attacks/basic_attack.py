from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from torchvision import transforms
import PIL.Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class basic_attacker():
    def __init__(self, config: dict, test: bool) -> None:
        self.target = config['target']
        self.rate = config['rate']
        self.clean = config['clean']
        self.name = 'basic'
        self.test = test
        self.channel_num = 3
        self.label_num = 10
        if self.test:
            self.rate = 1.

    def make_trigger(self, data):
        raise NotImplementedError(
            'You haven\'t achieve {} method!'.format(self.name))

    def data_volume_regular(self, dataset: Dataset):
        data = dataset
        self.remain_length = len(data)
        remain_data = list(data[i][0] for i in range(len(data)))

        if hasattr(dataset, 'targets'):
            remain_label = list(dataset.targets)
        elif hasattr(dataset, 'labels'):
            remain_label = list(dataset.labels)

        self.target_num = remain_label.count(self.target)
        self.remain_dataset = list(zip(remain_data, remain_label))
        self.bd_volume = round(self.rate*self.remain_length)

    def data_random_picker(self):

        while True:
            index = random.randint(0, self.remain_length-1)
            data_packer = self.remain_dataset[index]
            img, label = data_packer

            cond1 = label == self.target
            cond2 = self.clean

            if not self.test:
                judge = ~ (cond1 ^ cond2)
            else:
                judge = ~ cond1

            if judge:
                break

        yield (img, label)

        posioned_data = self.make_trigger(img)

        del self.remain_dataset[index]
        self.remain_length = self.remain_length-1

        hard_label = self.target
        yield (posioned_data, hard_label)

    def make_bd_dataset(self):

        class bd_dataset(Dataset):
            def __init__(self, train: bool) -> None:
                super().__init__()
                self.data = []
                self.labels = []

                if train:
                    self.transform = transforms.Compose([transforms.Resize(32),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.RandomCrop(
                                                             32, 4),
                                                         transforms.ToTensor()])

                else:
                    self.transform = transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.ToTensor()
                        ]
                    )

            def add_packer(self, packer):
                self.data.append(packer[0])
                self.labels.append(packer[1])

            def __getitem__(self, index):
                data = self.data[index]
                data = self.transform(data)
                label = self.labels[index]
                if isinstance(label, int):
                    label = torch.tensor(label)
                return data, label

            def fusion(self, remain: list):
                length = len(remain)
                self.data = self.data + [remain[i][0] for i in range(length)]
                self.labels = self.labels + [remain[i][1]
                                             for i in range(length)]

            def __len__(self):
                return len(self.data)

        self.bd_dataset = bd_dataset(train=~self.test)

    def make_train_bddataset(self, dataset: Dataset):
        self.data_volume_regular(dataset)
        self.make_bd_dataset()

        for idx in range(self.bd_volume):
            data_breeder = self.data_random_picker()

            clean_packer = data_breeder.__next__()
            self.bd_dataset.add_packer(packer=clean_packer)

            poisoned_packer = data_breeder.__next__()
            self.bd_dataset.add_packer(packer=poisoned_packer)

        self.bd_dataset.fusion(self.remain_dataset)
        return self.bd_dataset

    def make_test_bddataset(self, dataset: Dataset):
        self.data_volume_regular(dataset)
        self.make_bd_dataset()

        while True:

            data_breeder = self.data_random_picker()

            data_breeder.__next__()
            poisoned_packer = data_breeder.__next__()
            self.bd_dataset.add_packer(packer=poisoned_packer)
            if self.remain_length == self.target_num:
                break

        return self.bd_dataset

    def data_sequtial_picker(self):
        index = 0
        while True:
            data_packer = self.remain_dataset[index]
            img, label = data_packer

            cond1 = label == self.target
            cond2 = self.clean

            if not self.test:
                judge = ~ (cond1 ^ cond2)
            else:
                judge = ~ cond1

            index += 1

            if judge:
                break

        yield (img, label)

        posioned_data = self.make_trigger(img)

        del self.remain_dataset[index]
        self.remain_length = self.remain_length-1

        hard_label = self.target
        yield (posioned_data, hard_label)

    def make_test_bddataset_sequential(self, dataset: Dataset):
        self.data_volume_regular(dataset)
        self.make_bd_dataset()

        while True:

            data_breeder = self.data_sequtial_picker()
            data_breeder.__next__()
            poisoned_packer = data_breeder.__next__()
            self.bd_dataset.add_packer(packer=poisoned_packer)
            if self.remain_length == self.target_num:
                break

        return self.bd_dataset
