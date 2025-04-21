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
    from brokenaxes import brokenaxes
    from matplotlib.gridspec import GridSpec

    class Range(_TRAIN):
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
            model.eval()

            bd_trainloader, bd_testloader, testloader = self.make_loader(
                bd_trainset, bd_testset, testset)
            bd_testloader = iter(bd_testloader)
            loss_func = nn.CrossEntropyLoss()
            layers = list(model.layers)
            layers_ = []
            for layer in layers:
                if 'relu' in layer:
                    layers_.append(layer)
            layers = layers_

            layer_wise_ori = {layers[i]: []
                              for i in range(len(layers))}
            layer_wise_bd = {layers[i]: []
                             for i in range(len(layers))}
            n = 0
            for _, (images, labels) in tqdm(enumerate(testloader), desc='Discrepency', colour='blue', total=round(len(testloader))):

                if labels == 5:
                    continue
                else:
                    n += 1
                    if n >= 0.1*len(testloader):
                        break
                    ori_img = images.cuda()
                    bd_img = bd_testloader.__next__()[0].cuda()
                    ori_features = model.forward(ori_img)
                    bd_features = model.forward(bd_img)

                    for layer in layers:

                        ori_layer_wise_features = ori_features[layer].view(
                            1, -1).detach().cpu()
                        bd_layer_wise_features = bd_features[layer].view(
                            1, -1).detach().cpu()

                        layer_wise_ori[layer].append(
                            ori_layer_wise_features.numpy())
                        layer_wise_bd[layer].append(
                            bd_layer_wise_features.numpy())

            import matplotlib.pyplot as plt
            import time
            from numpy import linalg

            plt.figure(figsize=(14, 8))
            k = 0

            data_ori_full = {}
            data_bd_full = {}
            for layer in layers:
                if "relu" in layer:
                    k += 1
                    pass
                else:
                    continue

                MAX = 0
                for i in range(len(layer_wise_ori[layer])):
                    MAX += linalg.norm(layer_wise_ori[layer][i], ord=2)
                MAX /= len(layer_wise_ori[layer])

                data_ori = np.concatenate(layer_wise_ori[layer], axis=0)
                data_ori = np.reshape(data_ori, (1, -1)).squeeze()
                data_bd = np.concatenate(layer_wise_bd[layer], axis=0)
                data_bd = np.reshape(data_bd, (1, -1)).squeeze()

                data_ori_full[layer] = data_ori/MAX
                data_bd_full[layer] = data_bd/MAX
                print('{} is finished.'.format(layer))

            ori_statistic = {}
            bd_statistic = {}
            contrast_statistic_max = {}
            contrast_statistic_min = {}
            contrast_statistic_pop = {}
            contrast_statistic_color = {}

            for layer in layers:
                ori_statistic[layer] = []
                ori_statistic[layer].append(np.min(data_ori_full[layer]))
                ori_statistic[layer].append(
                    np.mean(data_ori_full[layer][(data_ori_full[layer] > 0.02)]))

                bd_statistic[layer] = []
                bd_statistic[layer].append(np.min(data_bd_full[layer]))
                bd_statistic[layer].append(
                    np.mean(data_bd_full[layer][(data_bd_full[layer] > 0.02)]))

                contrast_statistic_max[layer] = []
                contrast_statistic_min[layer] = []
                contrast_statistic_max[layer].append(
                    np.max(data_ori_full[layer]-data_bd_full[layer]))
                contrast_statistic_min[layer].append(
                    np.min(data_ori_full[layer]-data_bd_full[layer]))

                contrast_statistic_pop[layer] = []
                contrast_statistic_color[layer] = []

                data = data_ori_full[layer]-data_bd_full[layer]
                for c in range(0, 100):
                    mask = np.logical_and(data < (np.min(
                        data)+(c+1)*(np.max(data)-np.min(data))/100), data >= (np.min(data)+(c)*(np.max(data)-np.min(data))/100))
                    contrast_statistic_pop[layer].append(
                        np.sum(mask)/np.size(data))

                    if np.min(
                            data)+(c+1)*(np.max(data)-np.min(data))/100 <= 0:
                        contrast_statistic_color[layer].append('r')
                    else:
                        contrast_statistic_color[layer].append('b')

            fontsize = 12
            bias = 0.125
            alpha = 0.5
            wspace = 0.3
            hspace = 2
            idx = 0
            for _, (layer) in enumerate(layers):
                if 'relu' in layer:
                    print(idx)
                    idx += 1
                    plt.subplot(4, 4, idx)
                    contrast_statistic_pop[layer] = np.array(
                        contrast_statistic_pop[layer])
                    max_item = np.max(contrast_statistic_pop[layer])
                    xaxis = range(0, 100)
                    plt.bar(xaxis, contrast_statistic_pop[layer],
                            bottom=0,
                            alpha=alpha, color=contrast_statistic_color[layer])
                    plt.xticks([])
                    plt.ylim(0, 0.3*max_item)
                    plt.yticks([])
                    plt.title('{}th layer'.format(idx), fontsize=fontsize)
                    plt.xlabel('$\Delta^{l}$',
                               fontsize=fontsize, labelpad=0.25)
                    plt.ylabel('$p(\Delta^{l})$',
                               fontsize=fontsize, labelpad=0.1)

            plt.subplots_adjust(wspace=0.2, hspace=0.4)

            plt.savefig(
                '/home/jyl/distill-defense/preliminary/density_{}.png'.format(attack), bbox_inches='tight')
            plt.savefig(
                '/home/jyl/distill-defense/preliminary/density_{}.pdf'.format(attack), bbox_inches='tight')

            plt.close()

    x = Range()
    x.train()
