import sys
sys.path.append('/home/jyl/distill-defense/utils/')
sys.path.append('/home/jyl/distill-defense/')

if __name__ == '__main__':

    from utils.picker import extractor_picker
    from main_config import *
    from generator import Generator
    from torch.optim import Adam, SGD
    from bn_parameter_getter import resnet18_bn_getter
    from tqdm import tqdm
    import torch.nn as nn

    model, extractor = extractor_picker(model_config)
    save_dir_path = model_config['save_path']
    load_model_path = save_dir_path + '{}/{}/{}.pth'.format(
        dataset_config['dataset'], model_config['model'], auxiliary_config['attack'])

    model.load_state_dict(torch.load(load_model_path)['state_dict'])
    extractor.model.load_state_dict(torch.load(load_model_path)['state_dict'])

    model, extractor = model.to('cuda'), extractor.to('cuda')
    model.eval()
    extractor.eval()

    for p in model.parameters():
        p.requires_grad_(False)
    for p in extractor.model.parameters():
        p.requires_grad_(False)

    bn_running_mean, bn_running_var = resnet18_bn_getter(model)
    bn_num = len(bn_running_mean)

    layers = list(extractor.layers)

    approximiate_a = torch.rand((1024, 3, 32, 32)).to('cuda')
    approximiate_b = torch.rand((1024, 3, 32, 32)).to('cuda')
    features_a = extractor.forward(approximiate_a)
    features_b = extractor.forward(approximiate_b)
    bn_running_var_bias = []

    for layer in layers:
        if 'conv' in layer:
            B, C, H, W = features_a[layer].shape
            intermediate_output_a = features_a[layer].view(C, -1)
            intermediate_output_b = features_b[layer].view(C, -1)
            bn_running_var_bias.append(torch.mean(
                torch.var(intermediate_output_a-intermediate_output_b, dim=-1)))

    def bn_loss(reverse_data, extractor=extractor):
        features = extractor.forward(reverse_data)

        G_running_mean = []
        G_running_var = []

        for layer in layers:
            if 'conv' in layer:
                B, C, H, W = features[layer].shape
                intermediate_output = features[layer].view(C, -1)

                G_running_mean.append(intermediate_output.mean(dim=-1))
                G_running_var.append(intermediate_output.var(dim=-1))

        mean_loss = []
        var_loss = []
        all_loss = []

        for i in range(bn_num):
            mean_loss.append(torch.norm(
                bn_running_mean[i-1]-G_running_mean[i-1]).pow(2))
            var_loss.append(torch.norm(
                bn_running_var[i-1]-G_running_var[i-1]).pow(2))
            all_loss.append(
                torch.max(mean_loss[-1]+var_loss[-1], 0))

        return sum(mean_loss) + sum(var_loss)

    reverse_data = torch.randn(16, 3, 32, 32).clamp_(0., 1.).cuda()
    reverse_data.detach_().requires_grad_(True)

    label = torch.randint(0, 10, (16,)).cuda()
    extractor.eval()
    model.eval()
    for param in extractor.parameters():
        param.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)

    import matplotlib.pyplot as plt
    from torchvision import transforms
    import numpy as np
    unloader = transforms.ToPILImage()
    gaussian = transforms.GaussianBlur(3)
    ce_loss_func = nn.CrossEntropyLoss()
    opt = Adam([reverse_data], 1e-1, (0.5, 0.999))

    ce_loss_record = []
    bn_loss_record = []

    for _ in range(1000):
        print(_)
        opt.zero_grad()
        loss = bn_loss(reverse_data, extractor)
        tv_loss = torch.norm(reverse_data, p=1)*5e-3
        ce_loss = ce_loss_func(model(reverse_data), label)

        full_loss = 2.5*loss+1*ce_loss
        full_loss.backward()

        opt.step()

        reverse_data.detach_().clamp_(0., 1.).requires_grad_(True)

        # print('bn: {}'.format(loss.item()))
        # print('ce: {}'.format(ce_loss.item()))
        bn_loss_record.append(loss.item())
        ce_loss_record.append(ce_loss.item())

        for i in range(10):
            data = reverse_data[i].cpu().detach()
            img = unloader(data)
            img = gaussian(img)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            plt.savefig('/home/jyl/distill-defense/distill/img_{}.png'.format(i), bbox_inches='tight')
            plt.savefig('/home/jyl/distill-defense/distill/img_{}.pdf'.format(i), bbox_inches='tight')
            plt.close()