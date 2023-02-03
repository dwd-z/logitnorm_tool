import os
import torch
import torch.nn as nn

from models.wrn import WideResNet
from models.cnn import CNN
from torchvision.models import resnet18

def build_model(model_type, num_classes, device, load=True, path='./snapshots', filename='cifar10_wrn_normal_standard'):
    net = None
    if model_type == 'wrn':
        net = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3)
    elif model_type == 'cnn':
        net = CNN(num_classes)
    elif model_type == 'rn18':
        net = resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)

    net.to(device)
    
    start_epoch = 0
    if load:
        for i in range(1000 - 1, -1, -1):
            model_name = os.path.join(path, filename + '_epoch_' + str(i) + '.pt')
            if os.path.isfile(model_name):
                net.load_state_dict(torch.load(model_name, map_location=device))
                print('Model restored! Epoch:', i)
                start_epoch = i + 1
                break
        if start_epoch == 0:
            assert False, "could not resume"

    return net