import torch.nn as nn


def resnet18_bn_getter(model: nn.Module):
    bn_running_mean = []
    bn_running_var = []

    bn_running_mean.append(model.layer1[0].bn1.running_mean)
    bn_running_mean.append(model.layer1[0].bn2.running_mean)
    bn_running_mean.append(model.layer1[1].bn1.running_mean)
    bn_running_mean.append(model.layer1[1].bn2.running_mean)

    bn_running_mean.append(model.layer2[0].bn1.running_mean)
    bn_running_mean.append(model.layer2[0].bn2.running_mean)
    bn_running_mean.append(model.layer2[1].bn1.running_mean)
    bn_running_mean.append(model.layer2[1].bn2.running_mean)

    bn_running_mean.append(model.layer3[0].bn1.running_mean)
    bn_running_mean.append(model.layer3[0].bn2.running_mean)
    bn_running_mean.append(model.layer3[1].bn1.running_mean)
    bn_running_mean.append(model.layer3[1].bn2.running_mean)

    bn_running_mean.append(model.layer4[0].bn1.running_mean)
    bn_running_mean.append(model.layer4[0].bn2.running_mean)
    bn_running_mean.append(model.layer4[1].bn1.running_mean)
    bn_running_mean.append(model.layer4[1].bn2.running_mean)

    bn_running_var.append(model.layer1[0].bn1.running_var)
    bn_running_var.append(model.layer1[0].bn2.running_var)
    bn_running_var.append(model.layer1[1].bn1.running_var)
    bn_running_var.append(model.layer1[1].bn2.running_var)

    bn_running_var.append(model.layer2[0].bn1.running_var)
    bn_running_var.append(model.layer2[0].bn2.running_var)
    bn_running_var.append(model.layer2[1].bn1.running_var)
    bn_running_var.append(model.layer2[1].bn2.running_var)

    bn_running_var.append(model.layer3[0].bn1.running_var)
    bn_running_var.append(model.layer3[0].bn2.running_var)
    bn_running_var.append(model.layer3[1].bn1.running_var)
    bn_running_var.append(model.layer3[1].bn2.running_var)

    bn_running_var.append(model.layer4[0].bn1.running_var)
    bn_running_var.append(model.layer4[0].bn2.running_var)
    bn_running_var.append(model.layer4[1].bn1.running_var)
    bn_running_var.append(model.layer4[1].bn2.running_var)

    return bn_running_mean, bn_running_var
