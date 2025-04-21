import sys
sys.path.append('/home/jyl/distill-defense/utils/')
sys.path.append('/home/jyl/distill-defense/')

if __name__ == '__main__':

    from utils.picker import extractor_picker
    from main_config import *
    from generator import  Generator
    from torch.optim import Adam
    from bn_parameter_getter import resnet18_bn_getter
    from tqdm import tqdm
    import torch.nn as nn

    model, extractor = extractor_picker(model_config)
    save_dir_path = model_config['save_path']
    load_model_path = save_dir_path + '{}/{}/{}.pth'.format(dataset_config['dataset'], model_config['model'], auxiliary_config['attack'])

    model.load_state_dict(torch.load(load_model_path)['state_dict'])
    extractor.model.load_state_dict(torch.load(load_model_path)['state_dict'])

    model, extractor = model.to('cuda'), extractor.to('cuda')
    model.eval()
    extractor.eval()

    for p in model.parameters():
        p.requires_grad_(False)
    for p in extractor.model.parameters():
        p.requires_grad_(False)

    G = Generator().cuda()
    
    opt = Adam(G.parameters(), 1e-4, (0.5, 0.999))

    bn_running_mean, bn_running_var = resnet18_bn_getter(model)
    bn_num = len(bn_running_mean)

    layers = list(extractor.layers)

    batchsize = 128
    latent_dim = 100
    iter_max = 200
    epoch_max = 400
    
    ratio_guess = 0.1
    
    approximiate = torch.rand((1024, 3, 32, 32)).to('cuda')
    features = extractor.forward(2*approximiate-1)
    bn_running_mean_bias = []
    bn_running_var_bias = []

    for layer in layers:
        if 'conv' in layer:
            B, C, H, W = features[layer].shape
            intermediate_output = features[layer].view(C, -1)
            bn_running_mean_bias.append(intermediate_output.mean(dim=-1))
            bn_running_var_bias.append(intermediate_output.var(dim=-1))

    def bn_loss(output, extractor = extractor, strategy = 'mean'):
        features = extractor.forward(output)

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

        if strategy == 'mean':
            for i in range(bn_num):
                mean_loss.append(max(torch.norm(
                    bn_running_mean[i-1]-G_running_mean[i-1]).pow(2)/bn_num, 0.0*torch.norm(
                    bn_running_mean_bias[i-1]-G_running_mean[i-1]).pow(2)/bn_num))
                var_loss.append(max(torch.norm(
                    bn_running_var[i-1]-G_running_var[i-1]).pow(2)/bn_num, 0.0*torch.norm(
                    bn_running_var_bias[i-1]-G_running_var[i-1]).pow(2)/bn_num))
        
        elif strategy == 'ascend':
            for i in range(1, bn_num+1):
                mean_loss.append(i*max(torch.norm(
                    bn_running_mean[i-1]-G_running_mean[i-1]).pow(2)/bn_num, 0.9*torch.norm(
                    bn_running_mean_bias[i-1]-G_running_mean[i-1]).pow(2)/bn_num))
                var_loss.append(i*max(torch.norm(
                    bn_running_var[i-1]-G_running_var[i-1]).pow(2)/bn_num, 0.9*torch.norm(
                    bn_running_var_bias[i-1]-G_running_var[i-1]).pow(2)/bn_num))
                
        return sum(mean_loss), sum(var_loss)
    
    loss_func = nn.CrossEntropyLoss()
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import numpy as np
    unloader = transforms.ToPILImage()

    for epoch in range(1, epoch_max+1):
        print('********************************** Begin the {}th epoch ***************************************'.format(epoch))
        for iter in tqdm(range(1, iter_max+1), desc='G Training', colour='blue', total=iter_max):
            G.train()
            opt.zero_grad()
            z = torch.randn(batchsize, latent_dim).cuda().contiguous()
            label = torch.randint(0, 10, (batchsize,)).cuda()
            images = G.forward(z, label)
            output = model(images)
            bn_mean_loss, bn_var_loss = bn_loss(images, extractor)
            ce_loss = loss_func(output, label)
            l2_norm = torch.norm(images.view(batchsize, -1), dim=-1,p=2).mean()
            loss = bn_mean_loss + bn_var_loss + ce_loss + 5e-5*l2_norm
            loss.backward()
            opt.step()
        
        print(bn_mean_loss.item())
        print(bn_var_loss.item())
        print(ce_loss.item())
        print(l2_norm.item())

        torch.save(G.state_dict(), '/home/jyl/distill-defense/distill/G.pth')
        G.load_state_dict(torch.load('/home/jyl/distill-defense/distill/G.pth'))
        G.eval()
        z = torch.randn(10, latent_dim).cuda().contiguous()
        label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
        images = G.forward(z, label).detach()

        for i in range(10):
            img = unloader(images[i])
            plt.imshow(img)
            plt.savefig('generator_{}.png'.format(i+1))
            plt.close()





    


    
       