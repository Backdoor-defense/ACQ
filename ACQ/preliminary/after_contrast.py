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

            k = 0
            for _, (images, labels) in tqdm(enumerate(testloader), desc='Discrepency', colour='blue', total=round(0.1*len(testloader))):
                k += 1
                if k >= 0.1*len(testloader)+1:
                    break

                if labels == 5:
                    continue
                else:
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

            plt.figure(figsize=(14, 6))
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

            fontsize = 12
            bias_ = 0.65
            bias = 0.6
            p_bias = 0.01
            n_bias = 0.05

            b_alpha = 0.35
            r_alpha = 0.25
            for idx, (layer) in enumerate(layers):
                if idx == 1:
                    plt.bar(idx+1, contrast_statistic_max[layer][0]-0,
                            bottom=0,
                            alpha=b_alpha, color='b', label='Positive')

                    plt.bar(idx+1, contrast_statistic_min[layer][0]-0,
                            bottom=0, alpha=r_alpha, color='r', label='Negative')
                    plt.text(idx+bias_, contrast_statistic_max[layer][0]+p_bias, '{}'.format(
                        round(100*contrast_statistic_max[layer][0], 2)))
                    plt.text(
                        idx+bias, contrast_statistic_min[layer][0]-n_bias, '{}'.format(round(100*contrast_statistic_min[layer][0], 2)))

                else:
                    plt.bar(idx+1, contrast_statistic_max[layer][0]-0,
                            bottom=0, alpha=b_alpha, color='b')

                    plt.bar(idx+1, contrast_statistic_min[layer][0]-0, bottom=0,
                            alpha=r_alpha, color='r')
                    plt.text(
                        idx+bias_, contrast_statistic_max[layer][0]+p_bias, '{}'.format(round(100*contrast_statistic_max[layer][0], 2)))
                    plt.text(
                        idx+bias, contrast_statistic_min[layer][0]-n_bias, '{}'.format(round(100*contrast_statistic_min[layer][0], 2)))

            plt.yticks([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                       ['100%', '80%', '60%', '40%', '20%', '0%', '-20%', '-40%', '-60%', '-80%', '-100%'], fontsize=fontsize)
            plt.xticks([i for i in range(1, 17)], ['{}'.format(i)
                       for i in range(1, 17)], fontsize=fontsize)
            plt.grid(axis='y')
            plt.xlabel('-th Convolution Layer', fontsize=fontsize)
            plt.ylabel('Discrepency $\Delta_{l}$', fontsize=fontsize)
            ax = plt.gca()
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            plt.legend(fontsize=10)

            plt.savefig(
                '/home/jyl/distill-defense/preliminary/after_box_{}.png'.format(attack), bbox_inches='tight')
            plt.savefig(
                '/home/jyl/distill-defense/preliminary/after_box_{}.pdf'.format(attack), bbox_inches='tight')

            plt.close()

    x = Range()
    x.train()
