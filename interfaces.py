from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sklearn.metrics as metrics
import torch.optim as optim

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        '''
        Initialize a LogitNorm class which contains the loss function.

        Attributes
        ----------
        device : str
            The device on which to train the model. Can be 'cpu' or 'gpu'.
        t : float, optional
            Temperature parameter for logit normalization. The default is 1.0
        '''
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        '''
        This criterion performs logit nomarlization and calculate the cross entropy loss of normalized logits.

        Parameters
        ----------
        x : 2-D array of float or int
            The output logits - raw, unnormalized scores for each class - of the neural network.  
        target : array of int
            An array of true labels of the training dataset.
        
        Returns
        -------
        cross_entropy_loss : float
            The cross entropy loss between normalized x and target.
        '''
        # logit normalization with temperature scaling
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t=1.0):
        softmaxes = F.softmax(logits/t, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
        
def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    temperature = temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
    return logits / temperature


# This function probably should live outside of this class, but whatever
def get_optimal_temperature(model, valid_loader, device):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = _ECELoss().to(device)
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
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

def odin(inputs, outputs, model, temper, noiseMagnitude1, device):
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

def gradnorm(net, loader, ood_num_examples, device, in_dist=False, T=1.0, test_bs=200):
    _score = []

    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        if batch_idx >= ood_num_examples // test_bs and in_dist is False:
            break
        data = data.to(device)
        net.zero_grad()
        output = net(data)
        num_classes = output.shape[-1]
        targets = torch.ones((data.shape[0], num_classes)).to(device)
        output = output / T
        loss = torch.mean(torch.sum(-targets * logsoftmax(output), dim=-1))

        loss.backward()
        layer_grad = net.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        all_score = -layer_grad_norm
        _score.append(all_score)

    if in_dist:
        return np.array(_score).copy()
    else:
        return np.array(_score)[:ood_num_examples].copy()

def get_id_scores(net, test_loader, device, T=1.0, ood_score='MSP'):
    '''
    This function calculates the confidence scores for each prediction in the test dataset.

    Returns
    -------
    scores : array of float
        An array of confidence scores calculated from the loss function (OOD score function).
    right_scores : array of float
        An array of negative confidence scores calculated from the loss function which have lead to a correct
        prediction.
    wrong_scores : array of float
        An array of negative confidence scores calculated from the loss function which have lead to an incorrect
        prediction.

    '''
    _score = []
    _right_score = []
    _wrong_score = []

    if ood_score == 'gradnorm':
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
        
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    with torch.no_grad():
        
        for batch_idx, examples in enumerate(test_loader):
            data, target = examples[0], examples[1]
            data = data.to(device)
            if ood_score == 'Odin':
                data = Variable(data, requires_grad = True)
            
            if ood_score == 'gradnorm':
                net.zero_grad()
                output = net(data)
                num_classes = output.shape[-1]
                targets = torch.ones((data.shape[0], num_classes)).to(device)
                output = output / T
                loss = torch.mean(torch.sum(-targets * logsoftmax(output), dim=-1))

                loss.backward()
                layer_grad = net.fc.weight.grad.data

                layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
                all_score = -layer_grad_norm
                _score.append(all_score)

            else: 
                output = net(data)
                smax = to_np(F.softmax(output, dim=1))
                
                if ood_score == 'energy':
                    all_score = -to_np(T * torch.logsumexp(output / T, dim=1))
                    _score.append(all_score)

                elif ood_score == 'Odin':
                    odin_score = odin(data, output, net, T, noiseMagnitude1=0, device=device)
                    all_score = -np.max(to_np(F.softmax(output/T, dim=1)), axis=1)
                    _score.append(-np.max(odin_score, 1))
                else:
                    all_score = -np.max(to_np(F.softmax(output/T, dim=1)), axis=1)
                    _score.append(all_score)


            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(all_score[right_indices])
            _wrong_score.append(all_score[wrong_indices])

        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
  

def get_ood_scores(net, test_loader, ood_num_examples, device, T=1.0, ood_score='MSP', test_bs=200):
    '''
    This function calculates the confidence scores for each prediction in the test dataset.

    Returns
    -------
    scores : array of float
        An array of confidence scores calculated from the loss function (OOD score function).
    right_scores : array of float
        An array of negative confidence scores calculated from the loss function which have lead to a correct
        prediction.
    wrong_scores : array of float
        An array of negative confidence scores calculated from the loss function which have lead to an incorrect
        prediction.

    '''
    _score = []

    if ood_score == 'gradnorm':
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    with torch.no_grad():
        
        for batch_idx, examples in enumerate(test_loader):
            data, target = examples[0], examples[1]
            if batch_idx >= ood_num_examples // test_bs:
                break

            data = data.to(device)
            if ood_score == 'Odin':
                data = Variable(data, requires_grad = True)

            if ood_score == 'gradnorm':
                net.zero_grad()
                output = net(data)
                num_classes = output.shape[-1]
                targets = torch.ones((data.shape[0], num_classes)).to(device)
                output = output / T
                loss = torch.mean(torch.sum(-targets * logsoftmax(output), dim=-1))

                loss.backward()
                layer_grad = net.fc.weight.grad.data

                layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
                all_score = -layer_grad_norm
                _score.append(all_score)

            else: 
                output = net(data)

                if ood_score == 'energy':
                    all_score = -to_np(T * torch.logsumexp(output / T, dim=1))
                    _score.append(all_score)

                elif ood_score == 'Odin':
                    odin_score = odin(data, output, net, T, noiseMagnitude1=0, device=device)
                    all_score = -np.max(to_np(F.softmax(output / T, dim=1)), axis=1)
                    _score.append(-np.max(odin_score, 1))
                else:
                    all_score = -np.max(to_np(F.softmax(output / T, dim=1)), axis=1)
                    _score.append(all_score)


    return concat(_score)[:ood_num_examples].copy()

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[
        fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    # , fps[cutoff]/(fps[cutoff] + tps[cutoff])
    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def get_measures(_pos, _neg, recall_level=0.95, to_string=True, method_name=''):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = metrics.roc_auc_score(labels, examples)
    aupr = metrics.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level=recall_level)

    if to_string: 
        print_measures(auroc, aupr, fpr, method_name)

    return auroc, aupr, fpr

def get_calibration_scores(net, loader, device,  T=1.0, n_bins=15, to_string=True):
    '''
    This function calculates the ECE error of the neural network given a test dataset.

    '''
    logits_list = []
    labels_list = []

    ece_criterion = _ECELoss(n_bins=n_bins)
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]

            data = data.to(device)
            label = target.to(device)
            logits = net(data)

            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
    ece_error = ece_criterion(logits, labels, T)
    
    if to_string:
        print('\n\nECE Error')
        print('ECE Error {:.2f}'.format(100 * ece_error.item()))

    return ece_error

def print_measures(auroc, aupr, fpr, method_name='', recall_level=0.95):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(aurocs, auprs, fprs, method_name='', recall_level=0.95):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 *
          recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 *
          np.mean(aurocs), 100 * np.std(aurocs)))
    print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 *
          np.mean(auprs), 100 * np.std(auprs)))

def get_error_rate(right_score, wrong_score, to_string=True):

        num_right = len(right_score)
        num_wrong = len(wrong_score)

        error_rate = 100 * num_wrong / (num_wrong + num_right)

        if to_string:
            print('Error Rate {:.2f}'.format(error_rate))

        return error_rate

from datasets.utils import build_dataset

def get_ood_dataloaders(OOD_data_list, input_size, input_channels, mean, std, test_bs=200, prefetch_threads=4):
    ood_loader_dict = dict()
    for data_name in OOD_data_list:
        # load OOD test dataset
        ood_data = build_dataset(data_name, mode="test", size=input_size, channels=input_channels,
                                    mean=mean, std=std)
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                                num_workers=prefetch_threads, pin_memory=True)
        ood_loader_dict[data_name] = ood_loader
    return ood_loader_dict
    
def get_id_measures(net, test_loader, device, T, ood_score,
                    recall_level=0.95, to_string=True, method_name=''):

    # /////////////// Detection Prelims ///////////////
    in_score, right_score, wrong_score = get_id_scores(net, test_loader, device, T=T, ood_score=ood_score)

    # /////////////// Error Detection ///////////////
    error_rate = get_error_rate(right_score, wrong_score, to_string=to_string)
    # calculate the FPR95, AUROC, AUFR scores of the test result
    auroc, aupr, fpr = get_measures(-right_score, -wrong_score, recall_level=recall_level, 
                                        to_string=to_string, method_name=method_name)

    # /////////////// ECE Detection ///////////////
    ece_error = get_calibration_scores(net, test_loader, device, T=T, n_bins=15, to_string=to_string)

    return error_rate, ece_error, auroc, aupr, fpr

def get_ood_measures(net, test_loader, ood_loader_dict, ood_num_examples, device, T, ood_score,
                     recall_level=0.95, test_bs=200, num_to_avg=10, to_string=True, method_name=''):
    auroc_list, aupr_list, fpr_list = [], [], []
    
    # calculate the confidence scores of each ID data prediction
    in_score, right_score, wrong_score = get_id_scores(net, test_loader, device, T=T, ood_score=ood_score)
                                          
    for (data_name, ood_loader) in ood_loader_dict.items():
        
        if to_string:
            print('\n\n{} Detection'.format(data_name))

        aurocs, auprs, fprs = [], [], []
        for _ in range(num_to_avg):
            # calculate the confidence scores of each OOD data prediction
            out_score = get_ood_scores(net, ood_loader, ood_num_examples, device, T=T, ood_score=ood_score, 
                                        test_bs=test_bs)   
            # calculate the FPR95, AUROC, AUFR scores concerning differentiating ID and OOD data
            measures = get_measures(-in_score, -out_score, recall_level=recall_level, 
                                                    to_string=False, method_name=method_name)
            aurocs.append(measures[0])
            auprs.append(measures[1])
            fprs.append(measures[2])
        auroc = np.mean(aurocs)
        aupr = np.mean(auprs)
        fpr = np.mean(fprs)

        if to_string:
            if num_to_avg >= 5:
                print_measures_with_std(aurocs, auprs, fprs, method_name)
            else:
                print_measures(auroc, aupr, fpr, method_name)

        auroc_list.append(auroc)
        aupr_list.append(aupr)
        fpr_list.append(fpr)

    if to_string:
        # print mean results
        print('\n\nMean Test Results')
        print_measures(np.mean(auroc_list), np.mean(aupr_list),
                    np.mean(fpr_list), method_name=method_name)
    
    return auroc_list, aupr_list, fpr_list

def get_all_measures(net, test_loader, ood_loader_dict, ood_num_examples, device, T, ood_score, 
                     recall_level=0.95, test_bs=200, to_string=True, method_name=''):
    error_rate, ece_error, auroc, aupr, fpr = get_id_measures(net, test_loader, device, T, ood_score,
                    recall_level=recall_level, to_string=to_string, method_name=method_name)
    auroc_list, aupr_list, fpr_list = get_ood_measures(net, test_loader, ood_loader_dict, ood_num_examples, device, T, ood_score,
                     recall_level=recall_level, test_bs=test_bs, to_string=to_string, method_name=method_name)
    
    return error_rate, ece_error, auroc, aupr, fpr, auroc_list, aupr_list, fpr_list

from tqdm import tqdm
import abc

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

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]

def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

    """
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)

import time
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


if __name__ == "__main__":
    pass