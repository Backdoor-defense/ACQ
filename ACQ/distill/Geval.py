import sys
sys.path.append('/home/jyl/distill-defense/utils/')
sys.path.append('/home/jyl/distill-defense/')

if __name__ == '__main__':

    from utils.picker import extractor_picker
    from main_config import *
    from generator import Generator
    from torch.optim import Adam
    from bn_parameter_getter import resnet18_bn_getter
    from tqdm import tqdm
    import torch.nn as nn

    model, _ = extractor_picker(model_config)
    save_dir_path = model_config['save_path']
    load_model_path = save_dir_path + '{}/{}/{}.pth'.format(
        dataset_config['dataset'], model_config['model'], auxiliary_config['attack'])

    model.load_state_dict(torch.load(load_model_path)['state_dict'])

    model = model.to('cuda')
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    G = Generator().cuda().eval()

    batchsize = 128
    latent_dim = 100

    clean_num = 0
    batch_idx = 0
    G.load_state_dict(torch.load('/home/jyl/distill-defense/distill/G.pth'))

    finetune_data = []
    with torch.no_grad():
        for i in range(100):

            batch_idx += 1
            z = torch.randn(batchsize, latent_dim).cuda().contiguous()
            label = torch.randint(0, 10, (batchsize,)).cuda()
            images = G.forward(z, label)
            output = model(images)

            finetune_data.append(
                (images.clone().cpu().requires_grad_(False).numpy(), label.cpu().numpy()))

            pred = output.argmax(dim=1, keepdim=True)
            clean_num += pred.eq(label.view_as(pred)).sum().item()
            print(100*clean_num/(batch_idx*batchsize))

        save_finetune_path = save_dir_path + '{}/{}/{}_finetune.pth'.format(
            dataset_config['dataset'], model_config['model'], auxiliary_config['attack'])
    torch.save(finetune_data, save_finetune_path)
