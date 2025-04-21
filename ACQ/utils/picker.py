import torch
from torch import Tensor
from torchvision import datasets, transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Iterable, Callable
from attacks import *
from models import *


def dataset_picker(dataset_config: dict):

    dataset_name = dataset_config['dataset']
    dataset_save_path = dataset_config['dataset_save_path']
    dataset_size = dataset_config['size']
    train_batch_size = dataset_config['train_batch_size']
    test_batch_size = dataset_config['test_batch_size']

    if dataset_name == 'MNIST':

        trainset = datasets.MNIST(
            dataset_save_path,
            train=True,
            transform=transforms.Compose(
                [transforms.Grayscale(3), transforms.ToTensor(), transforms.ToPILImage(mode='RGB')])
        )

        testset = datasets.MNIST(
            dataset_save_path,
            train=False,
            transform=transforms.Compose(
                [transforms.Grayscale(3), transforms.ToTensor(), transforms.ToPILImage(mode='RGB')])
        )

    elif dataset_name == 'FashionMNIST':

        trainset = datasets.FashionMNIST(
            dataset_save_path,
            train=True,
            transform=transforms.Compose(
                [transforms.ToPILImage(), transforms.Grayscale(3)])
        )

        testset = datasets.FashionMNIST(
            dataset_save_path,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(3)])
        )

    elif dataset_name == 'CIFAR10':

        trainset = datasets.CIFAR10(
            dataset_save_path,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage()])
        )

        testset = datasets.CIFAR10(
            dataset_save_path,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage()])
        )

    elif dataset_name == 'SVHN':

        trainset = datasets.SVHN(
            '/home/data/SVHN',
            split='train',
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage()])
        )

        testset = datasets.SVHN(
            '/home/data/SVHN',
            split='test',
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage()])
        )

    elif dataset_name == 'CIFAR100':

        trainset = datasets.CIFAR100(
            dataset_save_path,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage()])
        )

        testset = datasets.CIFAR100(
            dataset_save_path,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage()])
        )

    elif dataset_name == 'ImageNet':

        trainset = datasets.ImageFolder(
            root=dataset_save_path,
        )

        testset = datasets.ImageFolder(
            root=dataset_save_path,
        )

    return {
        'trainset': trainset,
        'trainloader': DataLoader(trainset, train_batch_size, shuffle=True),
        'evalloader': DataLoader(trainset, test_batch_size),
        'train_number': len(trainset),
        'testset': testset,
        'testloader': DataLoader(testset, test_batch_size),
        'test_number': len(testset)
    }


def auxiliary_picker(auxiliary_config: dict, model: nn.Module):
    opt_name = auxiliary_config['opt']
    stl_name = auxiliary_config['stl']
    lr = auxiliary_config['lr']
    momentum = auxiliary_config['momentum']
    weight_decay = auxiliary_config['weight_decay']
    milestones = auxiliary_config['milestones']
    gamma = auxiliary_config['gamma']

    if opt_name == 'SGD' and stl_name == 'MultiStepLR':
        opt = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        stl = MultiStepLR(
            opt,
            milestones=milestones,
            gamma=gamma
        )
    else:
        raise NotImplementedError('You haven\'t realize this auxiliary!')

    return {
        'opt': opt,
        'stl': stl
    }


def model_picker(model_config: dict):
    num_classes = model_config['num_classes']
    model_name = model_config['model']

    if model_name == 'resnet18':
        model = resnet.ResNet18(num_classes)

    elif model_name == 'resnet34':
        model = resnet.ResNet34(num_classes)

    elif model_name == 'resnet50':
        model = resnet.ResNet50(num_classes)

    elif model_name == 'resnet101':
        model = resnet.ResNet101(num_classes)

    elif model_name == 'resnet152':
        model = resnet.ResNet152(num_classes)

    elif model_name == 'wideresnet':
        model = wideresnet.wideResnet28_10(num_classes)

    return model


def extractor_picker(model_config: dict):
    model = model_picker(model_config)
    layers = dict(model.named_modules()).keys()

    class FeatureExtractor(nn.Module):

        def __init__(self, model: nn.Module, layers: Iterable[str]):
            super().__init__()
            self.model = model
            self.layers = layers
            self._features = {layer: torch.empty(0) for layer in layers}

            for layer_id in layers:
                layer = dict([*self.model.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_outputs_hook(layer_id))

        def save_outputs_hook(self, layer_id: str) -> Callable:
            def fn(_, __, output):
                self._features[layer_id] = output
            return fn

        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            _ = self.model(x)
            features = {layer: self._features[layer].clone()
                        for layer in self.layers}
            return features

    extractor = FeatureExtractor(model, layers)
    return model, extractor


def attack_picker(attacker_config: dict):

    if attacker_config['attack'] == 'Badnets':

        config = {
            'target': 5,
            'rate': 0.1,
            'clean': False
        }

        train_attack = badnets.Badnets(config, False)
        test_attack = badnets.Badnets(config, True)

    elif attacker_config['attack'] == 'Blend':

        config = {
            'target': 5,
            'rate': 0.1,
            'clean': False
        }

        train_attack = blend.Blend(config, False)
        test_attack = blend.Blend(config, True)

    elif attacker_config['attack'] == 'SIG':

        config = {
            'target': 5,
            'rate': 0.1,
            'clean': False
        }
        train_attack = sig.SIG(config, False)
        test_attack = sig.SIG(config, True)

    elif attacker_config['attack'] == 'Trojan':

        config = {
            'target': 5,
            'rate': 0.1,
            'clean': False
        }
        train_attack = trojan.Trojan(config, False)
        test_attack = trojan.Trojan(config, True)

    elif attacker_config['attack'] == 'CL':

        config = {
            'target': 5,
            'rate': 0.1,
            'clean': False
        }
        train_attack = cl.CL(config, False)
        test_attack = cl.CL(config, True)

    return train_attack, test_attack


trainset = datasets.MNIST(
    '/home/data/',
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(3)])
)
print(trainset[0][0].size)
