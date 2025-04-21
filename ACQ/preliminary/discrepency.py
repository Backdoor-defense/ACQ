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
            model.eval()

            bd_trainloader, bd_testloader, testloader = self.make_loader(
                bd_trainset, bd_testset, testset)
            bd_testloader = iter(bd_testloader)
            loss_func = nn.CrossEntropyLoss()
            layers = list(model.layers)

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

            density_layer = {}

            plt.figure(figsize=(14, 8))
            k = 0

            prop_full = []
            data_full = []
            for layer in layers:
                if "relu" in layer:
                    k += 1
                    pass
                else:
                    continue

                descrepency = []
                MAX = 0
                for i in range(len(layer_wise_ori[layer])):
                    descrepency.append(
                        layer_wise_bd[layer][i]-layer_wise_ori[layer][i])
                    MAX += linalg.norm(layer_wise_ori[layer][i], ord=2)
                MAX /= len(layer_wise_ori[layer])

                descrepency = np.concatenate(descrepency, axis=0)

                data = descrepency
                data = data.transpose(1, 0)

                data = np.mean(data, axis=1)/MAX
                prop_full.append(max(np.max(data)/MAX, np.min(data)/MAX))
                data_full.append(data)
                print('{} is finished.'.format(layer))

            def get_median(data):
                data = sorted(list(data))
                size = len(data)
                if size % 2 == 0:
                    median = (data[size//2] + data[size//2 - 1])/2
                else:
                    median = data[(size-1)//2]
                return median

            median = []
            up_quarter = []
            down_quarter = []
            max_ = []
            min_ = []
            dense_first_quarter = []
            dense_second_quarter = []
            dense_third_quarter = []
            dense_forth_quarter = []

            for data in data_full:
                median.append(get_median(data))
                up_quarter.append(get_median(data[(data > median[-1])]))
                down_quarter.append(get_median(data[(data <= median[-1])]))
                max_.append(np.max(data))
                min_.append(np.min(data))

                dense_first_quarter.append(
                    len(data[(data > up_quarter[-1])])/len(data))
                dense_second_quarter.append(
                    len(data[np.logical_and(median[-1] < data, data <= up_quarter[-1])])/len(data))
                dense_third_quarter.append(
                    len(data[np.logical_and(down_quarter[-1] < data, data <= median[-1])])/len(data))
                dense_forth_quarter.append(
                    len(data[(data <= down_quarter[-1])])/len(data))

            # for i in range(len(median)):
            #     max_[i] = max_[i] - up_quarter[i]
            #     up_quarter[i] = up_quarter[i] - median[i]
            #     median[i] = median[i] - down_quarter[i]
            #     down_quarter[i] = down_quarter[i] - min_[i]
            plt.subplot(211)
            fontsize = 10
            xs = range(1, 17)
            for xs_ in xs:
                if xs_ == 1:
                    plt.bar(xs[xs_-1], down_quarter[xs_-1]-min_[xs_-1], bottom=min_[xs_-1],
                            alpha=dense_forth_quarter[xs_-1], color='b', label='The forth quartile')
                    plt.bar(xs[xs_-1], max_[xs_-1]-up_quarter[xs_-1], bottom=up_quarter[xs_-1],
                            alpha=dense_first_quarter[xs_-1], color='r', label='The first quartile')
                    plt.bar(xs[xs_-1], up_quarter[xs_-1]-median[xs_-1], bottom=median[xs_-1],
                            alpha=dense_second_quarter[xs_-1], color='y', label='The second quartile')
                    plt.bar(xs[xs_-1], median[xs_-1]-down_quarter[xs_-1], bottom=down_quarter[xs_-1],
                            alpha=dense_third_quarter[xs_-1], color='g', label='The third quartile')
                else:
                    plt.bar(xs[xs_-1], down_quarter[xs_-1]-min_[xs_-1], bottom=min_[xs_-1],
                            alpha=dense_forth_quarter[xs_-1], color='b')
                    plt.bar(xs[xs_-1], max_[xs_-1]-up_quarter[xs_-1], bottom=up_quarter[xs_-1],
                            alpha=dense_first_quarter[xs_-1], color='r')
                    plt.bar(xs[xs_-1], up_quarter[xs_-1]-median[xs_-1], bottom=median[xs_-1],
                            alpha=dense_second_quarter[xs_-1], color='y')
                    plt.bar(xs[xs_-1], median[xs_-1]-down_quarter[xs_-1], bottom=down_quarter[xs_-1],
                            alpha=dense_third_quarter[xs_-1], color='g')

            plt.yticks([0.3, 0.2, 0.1, 0., -0.1],
                       ['30%', '20%', '10%', '0%', '-10%'], fontsize=fontsize)
            plt.xticks([i for i in range(1, 17)], ['{}'.format(i)
                       for i in range(1, 17)], fontsize=fontsize)
            plt.xlabel('-th Convolution Layer', fontsize=fontsize)
            plt.ylabel('Discrepency', fontsize=fontsize)
            ax = plt.gca()
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            plt.legend(fontsize=10)

            plt.subplot(212)
            celltext = [
                ['{}%'.format(round(100*max_[i], 3)) for i in range(0, 16)],
                ['{}%'.format(round(100*up_quarter[i], 3))
                 for i in range(0, 16)],
                ['{}%'.format(round(100*median[i], 3)) for i in range(0, 16)],
                ['{}%'.format(round(100*down_quarter[i], 3))
                 for i in range(0, 16)],
                ['{}%'.format(round(100*min_[i], 3)) for i in range(0, 16)]]

            col = ['Layer{}'.format(i) for i in range(1, 17)]
            row = ['The forth quartile',
                   'The first quartile',
                   'The second quartile',
                   'The third quartile',
                   'Min']

            tab = plt.table(
                cellText=celltext,
                colLabels=col,
                rowLabels=row,
                cellLoc='center',
                rowLoc='center',
                loc='upper center',
            )
            tab.axes.axis('off')
            tab.auto_set_font_size(False)
            tab.set_fontsize(fontsize)
            tab.scale(1, 2)
            plt.subplots_adjust(hspace=0)
            plt.savefig(
                '/home/jyl/distill-defense/preliminary/box_{}.png'.format(attack))
            plt.savefig(
                '/home/jyl/distill-defense/preliminary/box_{}.pdf'.format(attack))

            plt.close()

    Pre = Discrepency()
    Pre.train()
