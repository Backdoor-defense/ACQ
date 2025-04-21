from torchvision import datasets, transforms
import numpy as np
import torchvision
import torch.autograd.functional as func
import torch
import numpy
dataset_config = {
    'dataset': 'CIFAR10',
    'dataset_save_path': '/home/data/',
    'size': [32, 32],
    'train_batch_size': 128,
    'test_batch_size': 1
}

model_config = {
    'model': 'resnet18',
    'num_classes': 10,
    'p': numpy.inf,
    'save_path': '/home/jyl/distill-defense/checkpoints/',
    'epoch_num': 40
}

auxiliary_config = {
    'opt': 'SGD',
    'stl': 'MultiStepLR',

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,

    'milestones': [25, 35],
    'gamma': 0.1,

    'attack': 'Badnets'
}
