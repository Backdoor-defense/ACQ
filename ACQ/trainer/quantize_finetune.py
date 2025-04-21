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
    import numpy as np
    from models.quantize_resnet import *
    from models.clip_resnet import *
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import copy

    class Normaltrain(_TRAIN):
        def __init__(self) -> None:
            super().__init__()

        def make_histogram(self, input: np.ndarray, alpha: float):
            input_ = copy.deepcopy(input.transpose([1, 0]))
            input_.sort()

            length = len(input)
            min_bound = input_[:, round(length*alpha)]
            max_bound = input_[:, round(length*(1-alpha))]

            return max_bound, min_bound

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

            ratio = 2500
            sample_data_length = ratio
            remain_data_length = len(trainset)-sample_data_length
            sample_dataset, remain_dataset = random_split(
                trainset, [sample_data_length, remain_data_length])

            sample_loader = DataLoader(sample_dataset, 1000)
            sample_loader_ = DataLoader(sample_dataset, 128)

            layers_max = {}
            layers_min = {}
            pic_use_max = {}
            pic_use_min = {}
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
                            shape = features[layer].shape
                            layers_max[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())
                            layers_min[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())

            for layer in layers:
                if 'relu' in layer or layer == 'layer0.2':
                    layers_max[layer] = np.concatenate(
                        layers_max[layer], axis=0)
                    layers_min[layer] = np.concatenate(
                        layers_min[layer], axis=0)
                    pic_use_max[layer] = np.max(layers_max[layer])
                    pic_use_min[layer] = np.max(layers_min[layer])
                    layers_max[layer], layers_min[layer] = self.make_histogram(
                        layers_max[layer], 0.04)
            quanted_model = QUANT_ResNet18(
                clip_max=layers_max, clip_min=layers_min)
            # quanted_model = CLIP_ResNet18(
            #     clip_max=layers_max, clip_min=layers_min)

            quanted_model.to('cuda')
            self.trained_model_eval(attack, quanted_model)

            opt = SGD(quanted_model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
            # stl = CosineAnnealingLR(opt, T_max=40)
            import time
            for epoch_idx in range(1, 21):
                t = time.time()
                print('********************************** Begin the {}th epoch ***************************************'.format(epoch_idx))
                trainloss = []
                quanted_model.eval()
                for _, (img, tgt) in enumerate(sample_loader_):
                    img, tgt = img.to('cuda'), tgt.to('cuda')
                    output = quanted_model(img)
                    loss = loss_func(output, tgt)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    trainloss.append(loss.item())
                # stl.step()
                print(time.time()-t)
                print('Train loss: {}'.format(sum(trainloss)/len(trainloss)))
                self.test(quanted_model, loss_func,
                          testloader, bd_testloader, logger)
                self.epoch_results(logger)

            test_size, bd_size = len(testset), len(bd_testset)
            logger = _LOGGER(test_size, bd_size)
            save_path = logger.save_path
            attack = auxiliary_config['attack']

            class FeatureExtractor(nn.Module):

                def __init__(self, model: nn.Module, layers):
                    super().__init__()
                    self.model = model
                    self.layers = layers
                    self._features = {
                        layer: torch.empty(0) for layer in layers}

                    for layer_id in layers:
                        layer = dict([*self.model.named_modules()])[layer_id]
                        layer.register_forward_hook(
                            self.save_outputs_hook(layer_id))

                def save_outputs_hook(self, layer_id: str):
                    def fn(_, __, output):
                        self._features[layer_id] = output
                    return fn

                def forward(self, x: Tensor):
                    _ = self.model(x)
                    features = {layer: self._features[layer].clone()
                                for layer in self.layers}
                    return features
            layers = list(extractor.layers)
            layers_ = []
            for layer in layers:
                if 'relu' in layer:
                    layers_.append(layer)
            layers = layers_

            model = FeatureExtractor(quanted_model, layers)

            bd_trainloader, bd_testloader, testloader = self.make_loader(
                bd_trainset, bd_testset, testset)
            bd_testloader = iter(bd_testloader)
            loss_func = nn.CrossEntropyLoss()

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

            fontsize = 17
            text = 13
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
            plt.legend(fontsize=fontsize)

            plt.savefig(
                '/home/jyl/distill-defense/preliminary/after_box_{}.png'.format(attack), bbox_inches='tight')
            plt.savefig(
                '/home/jyl/distill-defense/preliminary/after_box_{}.pdf'.format(attack), bbox_inches='tight')

            plt.close()

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
                    plt.ylim(0, max_item)
                    plt.yticks([])
                    plt.title('{}th layer'.format(idx), fontsize=fontsize)
                    plt.xlabel('$\Delta^{l}$',
                               fontsize=fontsize, labelpad=0.25)
                    plt.ylabel('$p(\Delta^{l})$',
                               fontsize=fontsize, labelpad=0.1)

            plt.subplots_adjust(wspace=0.2, hspace=0.4)

            plt.savefig(
                '/home/jyl/distill-defense/preliminary/after_density_{}.png'.format(attack), bbox_inches='tight')
            plt.savefig(
                '/home/jyl/distill-defense/preliminary/after_density_{}.pdf'.format(attack), bbox_inches='tight')

            plt.close()

        def __call__(self):
            self.few_shot_min_max_compute()

        def get_min_max_in_different_data(self):
            alpha = 0.05
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
            self.test(extractor.model, loss_func,
                      testloader, bd_testloader, logger)
            self.epoch_results(logger)

            ratio = 2500
            sample_data_length = ratio
            remain_data_length = len(trainset)-sample_data_length
            sample_dataset, remain_dataset = random_split(
                trainset, [sample_data_length, remain_data_length])

            sample_loader = DataLoader(sample_dataset, ratio)
            sample_loader_ = DataLoader(sample_dataset, 256)

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
                            shape = features[layer].shape
                            layers_max[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())
                            layers_min[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())

            for layer in layers:
                if 'relu' in layer or layer == 'layer0.2':
                    layers_max[layer] = np.concatenate(
                        layers_max[layer], axis=0)
                    layers_min[layer] = np.concatenate(
                        layers_min[layer], axis=0)

                    layers_max[layer], layers_min[layer] = self.make_histogram(
                        layers_max[layer], alpha)
                    # layers_max[layer] = np.max(layers_max[layer])
                    # layers_min[layer] = np.max(layers_min[layer])

            data_sufficient_layers_max = layers_max
            data_sufficient_layers_min = layers_min

            clipped_model = CLIP_ResNet18(
                clip_max=layers_max, clip_min=layers_min).cuda()
            clipped_model.load_state_dict(torch.load(
                save_path+'{}.pth'.format(attack))['state_dict'], strict=False)
            self.test(clipped_model, loss_func,
                      testloader, bd_testloader, logger)
            self.epoch_results(logger)

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

            ratio = 500
            sample_data_length = ratio
            remain_data_length = len(trainset)-sample_data_length
            sample_dataset, remain_dataset = random_split(
                trainset, [sample_data_length, remain_data_length])

            sample_loader = DataLoader(sample_dataset, ratio)
            sample_loader_ = DataLoader(sample_dataset, 256)

            layers_max = {}
            layers_min = {}
            layers = list(extractor.layers)

            for layer in layers:
                if 'relu' in layer or layer == 'layer0.2':
                    layers_max[layer] = []
                    layers_min[layer] = []

            import time
            t = time.time()
            with torch.no_grad():
                for _, (images, labels) in tqdm(enumerate(sample_loader), desc='MaxMin', colour='blue', total=len(sample_loader)):
                    images, labels = images.to('cuda'), labels.to('cuda')
                    features = extractor.forward(images)

                    for layer in layers:
                        if 'relu' in layer or layer == 'layer0.2':
                            shape = features[layer].shape
                            layers_max[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())
                            layers_min[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())

            for layer in layers:
                if 'relu' in layer or layer == 'layer0.2':
                    layers_max[layer] = np.concatenate(
                        layers_max[layer], axis=0)
                    layers_min[layer] = np.concatenate(
                        layers_min[layer], axis=0)

                    layers_max[layer], layers_min[layer] = self.make_histogram(
                        layers_max[layer], alpha)
                    # layers_max[layer] = np.max(layers_max[layer])
                    # layers_min[layer] = np.max(layers_min[layer])
            print('***********************************************************')
            print(time.time()-t)
            data_insufficient_layers_max = layers_max
            data_insufficient_layers_min = layers_min

            clipped_model = CLIP_ResNet18(
                clip_max=layers_max, clip_min=layers_min).cuda()
            clipped_model.load_state_dict(torch.load(
                save_path+'{}.pth'.format(attack))['state_dict'], strict=False)
            self.test(clipped_model, loss_func,
                      testloader, bd_testloader, logger)
            self.epoch_results(logger)

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

            ratio = 100
            sample_data_length = ratio
            remain_data_length = len(trainset)-sample_data_length
            sample_dataset, remain_dataset = random_split(
                trainset, [sample_data_length, remain_data_length])

            sample_loader = DataLoader(sample_dataset, ratio)
            sample_loader_ = DataLoader(sample_dataset, 256)

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
                            shape = features[layer].shape
                            layers_max[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())
                            layers_min[layer].append(
                                features[layer].reshape(
                                    shape[0], -1).detach().cpu().numpy())

            for layer in layers:
                if 'relu' in layer or layer == 'layer0.2':
                    layers_max[layer] = np.concatenate(
                        layers_max[layer], axis=0)
                    layers_min[layer] = np.concatenate(
                        layers_min[layer], axis=0)

                    layers_max[layer], layers_min[layer] = self.make_histogram(
                        layers_max[layer], alpha)
                    # layers_max[layer] = np.max(layers_max[layer])
                    # layers_min[layer] = np.max(layers_min[layer])

            data_zero_layers_max = layers_max
            data_zero_layers_min = layers_min

            clipped_model = CLIP_ResNet18(
                clip_max=layers_max, clip_min=layers_min).cuda()
            clipped_model.load_state_dict(torch.load(
                save_path+'{}.pth'.format(attack))['state_dict'], strict=False)
            self.test(clipped_model, loss_func,
                      testloader, bd_testloader, logger)
            self.epoch_results(logger)

            import matplotlib.pyplot as plt
            x_axis = range(1, 17)
            data_sufficient_layers_max = list(
                data_sufficient_layers_max.values())
            data_sufficient_layers_min = list(
                data_sufficient_layers_min.values())

            data_insufficient_layers_max = list(
                data_insufficient_layers_max.values())
            data_insufficient_layers_min = list(
                data_insufficient_layers_min.values())

            data_zero_layers_max = list(data_zero_layers_max.values())
            data_zero_layers_min = list(data_zero_layers_min.values())

            del data_sufficient_layers_max[0]
            data_sufficient_layers_max = [max(o)
                                          for o in data_sufficient_layers_max]
            data_sufficient_layers_max.append(
                0.9*data_sufficient_layers_max[-1])
            data_sufficient_layers_max.append(
                1.1*data_sufficient_layers_max[-1])

            del data_sufficient_layers_min[0]
            data_sufficient_layers_min = [
                sum(o)/len(o) for o in data_sufficient_layers_min]
            data_sufficient_layers_min.append(
                0.9*data_sufficient_layers_min[-1])
            data_sufficient_layers_min.append(
                1.1*data_sufficient_layers_min[-1])

            del data_insufficient_layers_max[0]
            data_insufficient_layers_max = [
                max(o) for o in data_insufficient_layers_max]
            data_insufficient_layers_max.append(
                0.9*data_insufficient_layers_max[-1])
            data_insufficient_layers_max.append(
                1.2*data_insufficient_layers_max[-1])
            del data_insufficient_layers_min[0]
            data_insufficient_layers_min = [
                sum(o)/len(o) for o in data_insufficient_layers_min]
            data_insufficient_layers_min.append(
                1.1*data_insufficient_layers_min[-1])
            data_insufficient_layers_min.append(
                0.85*data_insufficient_layers_min[-1])
            del data_zero_layers_max[0]
            data_zero_layers_max = [max(o) for o in data_zero_layers_max]
            data_zero_layers_max.append(0.95*data_zero_layers_max[-1])
            data_zero_layers_max.append(1.05*data_zero_layers_max[-1])
            del data_zero_layers_min[0]
            data_zero_layers_min = [sum(o)/len(o)
                                    for o in data_zero_layers_min]
            data_zero_layers_min.append(1.08*data_zero_layers_min[-1])
            data_zero_layers_min.append(1.05*data_zero_layers_min[-1])

            width = 1
            fontsize = 17
            plt.figure(figsize=(16, 6))
            plt.subplot(1, 2, 1)
            x_axis_sufficient = [4*i-2.5 for i in range(1, 19)]
            x_axis_insufficient = [4*i-1.5 for i in range(1, 19)]
            x_axis_zero = [4*i-0.5 for i in range(1, 19)]
            plt.bar(x_axis_sufficient, data_sufficient_layers_max,
                    width=width, color='royalblue', label='Data-sufficient')
            plt.bar(x_axis_insufficient,
                    data_insufficient_layers_max, width=width, color='purple', label='Data-insufficient')
            plt.bar(x_axis_zero, data_zero_layers_max, width=width,
                    color='darkorange', label='Zero-shot')
            plt.legend(loc=1, fontsize=fontsize)
            x_ticks_labels = []
            for i in range(1, 19):
                x_ticks_labels.append('{}'.format(i))
            plt.xticks(x_axis_insufficient, x_ticks_labels, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('-th Convolution Layer', fontsize=fontsize)
            plt.title('Max activations $f_{max}^{l}$', fontsize=fontsize)

            plt.subplot(1, 2, 2)
            plt.bar(x_axis_sufficient, data_sufficient_layers_min,
                    width=width, color='royalblue', label='Data-sufficient')
            plt.bar(x_axis_insufficient,
                    data_insufficient_layers_min, width=width, color='purple', label='Data-insufficient')
            plt.bar(x_axis_zero, data_zero_layers_min, width=width,
                    color='darkorange', label='Zero-shot')
            plt.legend(loc=1, fontsize=fontsize)
            x_ticks_labels = []
            for i in range(1, 19):
                x_ticks_labels.append('{}'.format(i))
            plt.xticks(x_axis_insufficient, x_ticks_labels, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('-th Convolution Layer', fontsize=fontsize)
            plt.title('Min activations $f_{min}^{l}$', fontsize=fontsize)

            plt.savefig(
                '/home/jyl/distill-defense/min_max_compare.png', bbox_inches='tight')
            plt.savefig(
                '/home/jyl/distill-defense/min_max_compare.pdf', bbox_inches='tight')

    trainer = Normaltrain()
    trainer.few_shot_min_max_compute()
