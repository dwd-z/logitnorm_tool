from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

from utils.loss import _ECELoss

def temperature_scale(logits, temperature):
    '''
    This function takes logits and performs temperature scaling on logits.

    Parameters
    ----------
    logits : array of float
        The raw output of the model before softmax score function.
    temperature : float
        The scaling to be imposed on the raw output, i.e. logits.

    Returns
    -------
    scaled_logits : array of float
        The scaled logits output of the model.

    '''
    # Expand temperature to match the size of logits
    temperature = temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
    return logits / temperature


# This function probably should live outside of this class, but whatever
def get_optimal_temperature(model, valid_loader, device, init=1.5):
    '''
    This function tunes the tempearature of the model using the validation dataset
    and minimizing negative log likelihood loss (NLL) as optimization goal.
    
    Parameters
    ----------
    valid_loader : torch.utils.data.DataLoader
        Data loader of the validation dataset.

    temperature : float
        The scaling to be imposed on the raw output, i.e. logits.

    Returns
    -------
    optimal_t : float
        The optimal temperature scale.

    '''
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = _ECELoss().to(device)
    temperature = nn.Parameter(torch.ones(1, device=device) * init)
    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, examples in enumerate(valid_loader):
            input, label = examples[0], examples[1]
            input = input.to(device)
            logits = model(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
    after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    return temperature.item()

def odin(inputs, outputs, model, temper, noiseMagnitude1, device): #TODO
    '''
    This function calculates the post-hoc odin scores for OOD detection given the input data 
    and raw output of the model.

    Parameters
    ----------
    inputs : array of float
        The raw output of the model before softmax score function.
    outputs : float
        The scaling to be imposed on the raw output, i.e. logits.
    model : nn.Module
        The neural network model. 
    temper : float 
        The temperature scale imposed on logits for odin scores calculation.
    noiseMagnitude1 : float
        Magnitude of small perturbations added to images.
    device : str
        Device on which to perform odin scores calculation.

    Returns
    -------
    odin_score : array of float
        odin scores for OOD detection

    '''
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs