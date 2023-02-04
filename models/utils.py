import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wrn import WideResNet
from models.cnn import CNN
from torchvision.models import resnet18

from tqdm import tqdm
import abc

from utils.loss import LogitNormLoss

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

class SingleModel:
    __metaclass__ = abc.ABCMeta
    def __init__(self, net, gpu, seed, loss_function, learning_rate, momentum, weight_decay, logitnorm_temp=1.0):

        self.gpu = gpu
        self.logitnorm_temp = logitnorm_temp
        self.net = net
        self.iterations = 0

        self.optimizer_model = torch.optim.SGD(self.net.parameters(), learning_rate,
                                                momentum=momentum, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_model, [80,140], gamma=0.1)

        if gpu is not None:
            device = torch.device('cuda:{}'.format(int(gpu)))
            torch.cuda.manual_seed(seed)
        else:
            device = torch.device('cpu')
        self.device = device

        if loss_function == "normal":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif loss_function == "logitnorm":
            self.loss_function = LogitNormLoss(device, self.logitnorm_temp)


    def train(self, train_loader):
        self.net.train()
        loss_avg = 0.0
        for data, target, index in tqdm(train_loader): # [128, 3, 32, 32], [128], [128]
            loss = self.train_batch(data, target)
            # backward
            self.optimizer_model.zero_grad()
            loss.backward()
            self.optimizer_model.step()
            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        self.scheduler.step()
        return loss_avg

    def train_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.net(inputs)
        loss = self.loss_function(logits, targets)
        return loss

    def test(self, test_loader):
        self.net.eval()
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for dict in test_loader:
                data, target = dict[0].to(self.device), dict[1].to(self.device)

                # forward
                output = self.net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        return loss_avg / len(test_loader), correct / len(test_loader.dataset)

def train_model(epochs, alg_obj, train_loader, test_loader, save_path='./snapshots/', method_name='', 
                save_weights=True, to_file=True, to_string=True):
    
    if to_file or save_weights:
        # Make save directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.isdir(save_path):
            raise Exception('%s is not a dir' % save_path)
    
    if to_file:
        # record the training results into a csv file
        with open(os.path.join(save_path, method_name +
                                        '_training_results.csv'), 'w') as f:
            f.write('epoch,time(s),train_loss,test_loss,test_acc(%)\n')
            print('Beginning Training\n')

    # Main loop
    state = dict()
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()
        state['train_loss'] = alg_obj.train(train_loader)
        state['test_loss'], state['test_accuracy'] = alg_obj.test(test_loader)

        # Save model
        if save_weights:
            torch.save(alg_obj.net.state_dict(),
                    os.path.join(save_path, method_name +
                                    '_epoch_' + str(epoch) + '.pt'))
            # delete the previous saved model
            prev_path = os.path.join(save_path, method_name +
                                    '_epoch_' + str(epoch - 1) + '.pt')
            if os.path.exists(prev_path): os.remove(prev_path)

        # Show results
        if to_file:
            with open(os.path.join(save_path, method_name +
                                            '_training_results.csv'), 'a') as f:
                f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                    (epoch + 1),
                    time.time() - begin_epoch,
                    state['train_loss'],
                    state['test_loss'],
                    100. * state['test_accuracy']
                ))
        if to_string:
            print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test acc {4:.2f}'.format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                state['train_loss'],
                state['test_loss'],
                100. * state['test_accuracy'])
            )