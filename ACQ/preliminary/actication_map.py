import sys
sys.path.append('/home/jyl/distill-defense/utils/')
sys.path.append('/home/jyl/distill-defense/')

if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    import torch.nn as nn
    from trainer._logger import _LOGGER
    from trainer._train import _TRAIN
    from main_config import dataset_config, model_config, auxiliary_config
    from utils.picker import *
    import numpy as np
    import matplotlib.pyplot as plt

    class Discrepency(_TRAIN):
        def __init__(self) -> None:
            super().__init__()

        def auxiliary_maker(self, auxiliary_picker: callable = auxiliary_picker,
                            model_picker: callable = extractor_picker,
                            dataset_picker: callable = dataset_picker):

            _, model = model_picker(self.model_config)
            dataset_packer = dataset_picker(self.dataset_config)
            auxiliary = auxiliary_picker(self.auxiliary_config, model)

            self.train_size = dataset_packer['train_number']
            self.test_size = dataset_packer['test_number']

            trainset = dataset_packer['trainset']
            testset = dataset_packer['testset']

            optimizer = auxiliary['opt']
            scheduler = auxiliary['stl']

            train_attacker, test_attacker = self.make_attacker()
            bd_trainset = train_attacker.make_train_bddataset(trainset)
            bd_testset = test_attacker.make_test_bddataset_sequential(testset)

            return model.to(self.device), bd_trainset, bd_testset, testset, optimizer, scheduler

        def make_loader(self, bd_trainset, bd_testset, testset):
            testset = self.testset_preprocess(testset)
            bd_trainloader = DataLoader(
                bd_trainset, self.dataset_config['train_batch_size'], shuffle=True)
            bd_testloader = DataLoader(bd_testset, batch_size=1)
            testloader = DataLoader(testset, batch_size=1)
            return bd_trainloader, bd_testloader, testloader

        def train(self):
            model, bd_trainset, bd_testset, testset, opt, stl = self.auxiliary_maker()

            test_size, bd_size = len(testset), len(bd_testset)
            logger = _LOGGER(test_size, bd_size)
            save_path = logger.save_path
            attack = auxiliary_config['attack']
            model.model.load_state_dict(torch.load(
                save_path+'{}.pth'.format(attack))['state_dict'])
            model.to('cuda')
            model.eval()

            bd_trainloader, bd_testloader, testloader = self.make_loader(
                bd_trainset, bd_testset, testset)
            bd_testloader = iter(bd_testloader)
            loss_func = nn.CrossEntropyLoss()
            layers = list(model.layers)
            layers = [layer for layer in layers if 'relu' in layer]

            layer_wise_output = {layer: [] for layer in layers}

            for _, (data, label) in enumerate(testloader):
                data, label = data.to('cuda'), label.to('cuda')
                features = model.forward(data)
                shape = data.shape
                for layer in layers:
                    layer_wise_output[layer].append(
                        features[layer].cpu().detach().view(shape[0], -1).numpy())

            for i, (layer) in enumerate(layers):
                print(layer)
                if 'relu' in layer:
                    layer_wise_output[layer] = np.concatenate(
                        layer_wise_output[layer], axis=0)
                    layer_wise_output[layer] = np.reshape(
                        layer_wise_output[layer], (1, -1)).squeeze()
                    plt.subplot(4, 4, i+1)
                    prop = []
                    max = np.max(layer_wise_output[layer])
                    min = np.min(layer_wise_output[layer])
                    for k in range(50):
                        feature = layer_wise_output[layer]
                        feature = feature[(feature > 0)]
                        length = len(feature)

                        if k == 0:
                            all_ = len(feature[np.logical_and(feature < (
                                min+(max-min)*(2*k+2)/100), 0. < feature)])
                            prop.append(
                                (all_)/length
                            )

                        else:
                            all_ = len(feature[np.logical_and(feature < (
                                min+(max-min)*(2*k+2)/100), (min+(max-min)*(2*k)/100) < feature)])
                            prop.append(
                                all_/length)

                    xaxis = range(1, 101, 2)
                    yaxis = prop
                    plt.bar(xaxis, yaxis)
                    plt.savefig(
                        '/home/jyl/distill-defense/preliminary/censity_{}.png'.format(attack))

    Pre = Discrepency()
    Pre.train()
