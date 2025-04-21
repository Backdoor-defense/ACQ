import sys
sys.path.append('/home/jyl/distill-defense/')
sys.path.append('/home/jyl/distill-defense/trainer/')


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    import torch.nn as nn
    from _logger import _LOGGER
    from _train import _TRAIN
    from utils import picker
    from main_config import *
    from torch.utils.data import random_split, DataLoader
    from torchvision import datasets, transforms
    import numpy as np
    from models.clip_resnet import*

    class Normaltrain(_TRAIN):
        def __init__(self) -> None:
            super().__init__()

        def train(self):
            model, bd_trainset, trainset, bd_testset, testset, opt, stl = self.auxiliary_maker()
            bd_trainloader, bd_testloader, testloader = self.make_loader(
                bd_trainset, bd_testset, testset)
            logger = _LOGGER(len(testset), len(bd_testset))
            loss_func = nn.CrossEntropyLoss()

            for epoch_idx in range(1, self.epoch_num + 1):

                print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))

                trainloss = []
                for _, (images, labels) in tqdm(enumerate(bd_trainloader), desc='Training', colour='blue', total=len(bd_trainloader)):
                    model.train()
                    images, labels = images.to(
                        self.device), labels.to(self.device)
                    opt.zero_grad()
                    output = self._impl_output(model, images)
                    loss = loss_func(output, labels)
                    trainloss.append(loss.item())

                    self.model_updater(opt, loss)

                self.test(model, loss_func, testloader, bd_testloader, logger)
                self.epoch_results(logger)
                self.checkpoints_save(logger)
                stl.step()

        def zero_shot_finetune(self):
            model, _, trainset, bd_testset, testset, opt, stl = self.auxiliary_maker()

            _, bd_testloader, testloader = self.make_loader(
                _, bd_testset, testset)

            save_finetune_path = '/home/jyl/distill-defense/checkpoints/CIFAR10/resnet18/Badnets_finetune.pth'
            trainloader = torch.load(save_finetune_path)

            logger = _LOGGER(len(testset), len(bd_testset))
            save_path = logger.save_path

            attack = 'Badnets'
            model.load_state_dict(torch.load(
                save_path+'{}.pth'.format(attack))['state_dict'])
            loss_func = nn.CrossEntropyLoss()

            for epoch_idx in range(1, self.epoch_num + 1):

                print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))

                trainloss = []
                for _, (images, labels) in tqdm(enumerate(trainloader), desc='Training', colour='blue', total=len(trainloader)):
                    model.eval()
                    images, labels = torch.tensor(images).to(
                        self.device), torch.tensor(labels).to(self.device)
                    opt.zero_grad()
                    output = self._impl_output(model, images)
                    loss = loss_func(output, labels)
                    trainloss.append(loss.item())

                    self.model_updater(opt, loss)

                self.test(model, loss_func, testloader, bd_testloader, logger)
                self.epoch_results(logger)
                stl.step()

        def few_shot_min_max_compute(self):
            _, bd_trainset, trainset, bd_testset, testset, opt, stl = self.auxiliary_maker()
            bd_trainloader, bd_testloader, testloader = self.make_loader(
                bd_trainset, bd_testset, testset)
            logger = _LOGGER(len(testset), len(bd_testset))
            loss_func = nn.CrossEntropyLoss()

            save_path = logger.save_path

            attack = auxiliary_config['attack']
            _, extractor = picker.extractor_picker(model_config)
            extractor.model.load_state_dict(torch.load(
                save_path+'{}.pth'.format(attack))['state_dict'], strict=False)
            extractor.eval()
            extractor.to('cuda')

            ratio = 25
            sample_data_length = ratio
            remain_data_length = len(trainset)-sample_data_length
            sample_dataset, remain_dataset = random_split(
                trainset, [sample_data_length, remain_data_length])

            sample_loader = DataLoader(sample_dataset, 128)

            layers_max = {}
            layers_min = {}
            layers = list(extractor.layers)

            for layer in layers:
                if 'relu' in layer or layer == 'layer0.2':
                    layers_max[layer] = []
                    layers_min[layer] = []

            with torch.no_grad():
                for _, (images, labels) in tqdm(enumerate(sample_loader), desc='MaxMin', colour='blue', total=len(sample_loader)):
                    images, labels = images.to('cuda'), labels.to('cuda')
                    features = extractor.forward(images)

                    for layer in layers:
                        if 'relu' in layer or layer == 'layer0.2':
                            layers_max[layer].append(
                                features[layer].detach().cpu().numpy())
                            layers_min[layer].append(
                                features[layer].detach().cpu().numpy())
            for layer in layers:
                if 'relu' in layer or layer == 'layer0.2':
                    layers_max[layer] = np.concatenate(
                        layers_max[layer], axis=0)
                    shape = layers_max[layer].shape
                    layers_max[layer] = np.reshape(
                        layers_max[layer], (shape[0], -1))
                    layers_min[layer] = np.concatenate(
                        layers_min[layer], axis=0)

                    shape = layers_min[layer].shape
                    layers_min[layer] = np.reshape(
                        layers_min[layer], (shape[0], -1))

                    mean = np.mean(layers_max[layer], axis=0, keepdims=True)
                    var = np.var(layers_max[layer], axis=0, keepdims=True)
                    var = np.sqrt(var, axes=0, keepdims=True)
                    layers_max[layer] = 0.8*np.max(layers_max[layer], axis=0)
                    layers_min[layer] = np.min(layers_min[layer], axis=0)

                    shape = list(shape)
                    shape[0] = 1
                    shape = tuple(shape)
                    layers_max[layer] = np.reshape(layers_max[layer], shape)
                    layers_min[layer] = np.reshape(layers_min[layer], shape)

            cliped_model = CLIP_ResNet18(
                clip_max=layers_max, clip_min=layers_min)
            cliped_model.to('cuda')
            self.trained_model_eval(attack, extractor.model)

        def finetune(self):
            model, bd_trainset, trainset, bd_testset, testset, opt, stl = self.auxiliary_maker()
            length = len(trainset)
            ratio = 0.01
            minidata = int(ratio*length)
            save_path = logger.save_path
            attack = auxiliary_config['attack']
            model.load_state_dict(torch.load(save_path+'{}.pth'.format(attack))['state_dict'], strict=False)
            trainset = random_split(trainset, [minidata, length-minidata])
            bd_trainloader, bd_testloader, testloader = self.make_loader(
                trainset, bd_testset, testset)
            logger = _LOGGER(len(testset), len(bd_testset))
            loss_func = nn.CrossEntropyLoss()

            for epoch_idx in range(1, 20 + 1):

                print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))

                trainloss = []
                for _, (images, labels) in tqdm(enumerate(bd_trainloader), desc='Training', colour='blue', total=len(bd_trainloader)):
                    model.train()
                    images, labels = images.to(
                        self.device), labels.to(self.device)
                    opt.zero_grad()
                    output = self._impl_output(model, images)
                    loss = loss_func(output, labels)
                    trainloss.append(loss.item())

                    self.model_updater(opt, loss)

                self.test(model, loss_func, testloader, bd_testloader, logger)
                self.epoch_results(logger)
                self.checkpoints_save(logger)
                stl.step()

            

        def __call__(self):
            self.train()

    trainer = Normaltrain()
    trainer()
