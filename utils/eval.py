from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sklearn.metrics as metrics

from utils.loss import _ECELoss
from utils.posthoc import odin, get_optimal_temperature


def get_confidence_scores(net, test_loader, device, T=1.0, score_function='MSP'):
    '''
    This function calculates the confidence scores for each prediction in the ID test dataset.

    Parameters
    ----------
    net : nn.Module
        The neural network model. 
    test_loader : torch.utils.data.DataLoader
        Data loader of the ID test dataset.
    device : str
        Device on which to perform confidence scores calculation.
    T : float, optional
        The post-hoc temperature scale for confidence scores calculation. The default is 1.0. 
    score_function : str, optional
        The score function used to calculate the confidence scores. The default is 'MSP'.

    Returns
    -------
    scores : array of float
        An array of confidence scores calculated from the score function.
    right_scores : array of float
        An array of negative confidence scores calculated from the MSP loss function which have lead to a correct
        prediction.
    wrong_scores : array of float
        An array of negative confidence scores calculated from the MSP loss function which have lead to an incorrect
        prediction.

    '''
    _right_score = []
    _wrong_score = []

    if score_function == 'gradnorm':
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
        
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    with torch.no_grad():
        
        for batch_idx, examples in enumerate(test_loader):
            data, target = examples[0], examples[1]
            data = data.to(device)
            if score_function == 'Odin':
                data = Variable(data, requires_grad = True)
            
            if score_function == 'gradnorm':
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

            else: 
                output = net(data)
                smax = to_np(F.softmax(output, dim=1))
                
                if score_function == 'energy':
                    all_score = -to_np(T * torch.logsumexp(output / T, dim=1))

                elif score_function == 'Odin':
                    all_score = -np.max(to_np(F.softmax(output/T, dim=1)), axis=1)
                else:
                    all_score = -np.max(to_np(F.softmax(output/T, dim=1)), axis=1)

            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(all_score[right_indices])
            _wrong_score.append(all_score[wrong_indices])

        return concat(_right_score).copy(), concat(_wrong_score).copy()
  

def get_ood_scores(net, test_loader, device, T=1.0, score_function='MSP', ood_num_examples=2000, test_bs=200):
    '''
    This function calculates the confidence scores for each prediction in the OOD test dataset.

    Parameters
    ----------
    net : nn.Module
        The neural network model. 
    test_loader : torch.utils.data.DataLoader
        Data loader of the OOD test dataset.
    device : str
        Device on which to perform confidence scores calculation.
    T : float, optional
        The post-hoc temperature scale for confidence scores calculation. The default is 1.0, 
    score_function : str, optional
        The score function used to calculate the confidence scores. The default is 'MSP'.
    ood_num_examples : int, optional
        Number of examples used to perform the test from the test dataset. The default is 2000.
    test_bs : int, optional
        The batch size of input to perform the test. The default is 200. 

    Returns
    -------
    scores : array of float
        An array of confidence scores calculated from the score function.

    '''
    _score = []

    if score_function == 'gradnorm':
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    with torch.no_grad():
        
        for batch_idx, examples in enumerate(test_loader):
            data, _ = examples[0], examples[1]
            if (ood_num_examples is not None) and (batch_idx >= ood_num_examples // test_bs):
                break

            data = data.to(device)
            if score_function == 'Odin':
                data = Variable(data, requires_grad = True)

            if score_function == 'gradnorm':
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

                if score_function == 'energy':
                    all_score = -to_np(T * torch.logsumexp(output / T, dim=1))
                    _score.append(all_score)

                elif score_function == 'Odin':
                    odin_score = odin(data, output, net, T, noiseMagnitude1=0, device=device)
                    all_score = -np.max(to_np(F.softmax(output / T, dim=1)), axis=1)
                    _score.append(-np.max(odin_score, 1))
                else:
                    all_score = -np.max(to_np(F.softmax(output / T, dim=1)), axis=1)
                    _score.append(all_score)

    return concat(_score)[:ood_num_examples].copy()


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    '''
    Use high precision for cumsum and check that final value matches sum.

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    '''
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    '''
    This function calculates the false positive rate (FPR) at a certain recall or true positive rate (TPR).

    Parameters
    ----------
    y_true : array of int
        An array of true labels.
    y_score : array of float
        An array of predicted scores.
    recall_level : float, optional
        The recall value or true positive rate (TPR) at which we calculate the false positive rate (FPR). 
        The default is 0.95. 
    pos_label : int, optional
        The label value for positive predictions. The default is None.

    Returns
    -------
    fpr : float
        The false positive rate (FPR) at a certain recall or true positive rate (TPR).

    '''
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

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def get_measures(_pos, _neg, recall_level=0.95, to_string=True, method_name=''):
    '''
    This function calculates the AUROC, AUPR and FPR at a certain recall.

    Parameters
    ----------
    _pos : array of float
        The scores from the model's positive prediction.
    _neg : array of float
        The scores from the model's negative prediction.
    recall_level : float, optional
        The recall value or true positive rate (TPR) at which we calculate the false positive rate (FPR). 
        The default is 0.95. 
    to_string : bool, optional
        If true, print out the AUROC, AUPR and FPR at a certain recall. The default is True.
    method_name : str, optional
        The method name of the model. The default is ''.

    Returns
    -------
    auroc : float
        The auroc score, i.e. the area under the receiver operating characteristic (AUROC).
    aupr : float
        The aupr score, i.e. the area under the precision-recall curve (AUPR). 
    fpr : float
        The fpr score at a certain recall, i.e. the false positive rate (FPR) at a certain recall.

    '''
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = metrics.roc_auc_score(labels, examples)
    aupr = metrics.average_precision_score(labels, examples)
    fpr = fpr_at_recall(labels, examples, recall_level=recall_level)

    if to_string: 
        print_measures(auroc, aupr, fpr, recall_level=recall_level, method_name=method_name)

    return auroc, aupr, fpr

def get_calibration_scores(net, test_loader, device, T=1.0, n_bins=15, to_string=True):
    '''
    This function calculates the expected calibration error (ECE) of the model given an ID test dataset.

    Parameters
    ----------
    net : nn.Module
        The neural network model. 
    test_loader : torch.utils.data.DataLoader
        Data loader of the ID test dataset.
    device : str
        Device on which to perform confidence scores calculation.
    T : float, optional
        The temperature scale for the ECE loss function. The default is 1.0.
    n_bins : int, optional
        The number of confidence interval bins for the ECE loss function. The default is 15.
    to_string : str, optional
        If true, print out the calculated ECE error. The default is True.

    Returns
    -------
    ece_error : float
        The expected calibration error (ECE) of the model given a test dataset.

    '''
    logits_list = []
    labels_list = []

    ece_criterion = _ECELoss(n_bins=n_bins)
    with torch.no_grad():
        for batch_idx, examples in enumerate(test_loader):
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


def print_measures(auroc, aupr, fpr, recall_level=0.95, method_name=''):
    '''
    This function prints out the performance measures of AUROC, AUPR and FPR at a certain recall.

    Parameters
    ----------
    auroc : float
        The auroc score, i.e. the area under the receiver operating characteristic (AUROC).
    aupr : float
        The aupr score, i.e. the area under the precision-recall curve (AUPR). 
    fpr : float
        The fpr score at a certain recall, i.e. the false positive rate (FPR) at a certain recall.
    recall_level : float, optional
        The recall value or true positive rate (TPR) at which we calculate the false positive rate (FPR). 
        The default is 0.95. 
    method_name : str, optional
        The method name of the model. The default is ''.

    Returns
    -------
    None.

    '''
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(aurocs, auprs, fprs, recall_level=0.95, method_name=''):
    '''
    This function prints out the mean performance measures of AUROC, AUPR and FPR at a certain recall
    with their standard deviations.

    Parameters
    ----------
    aurocs : array of float
        An array of auroc scores, i.e. the area under the receiver operating characteristic (AUROC).
    auprs : array of float
        An array of aupr scores, i.e. the area under the precision-recall curve (AUPR). 
    fprs : array of float
        An array of fpr scores at a certain recall, i.e. the false positive rate (FPR) at a certain recall.
    recall_level : float, optional
        The recall value or true positive rate (TPR) at which we calculate the false positive rates (FPR). 
        The default is 0.95. 
    method_name : str, optional
        The method name of the model. The default is ''.

    Returns
    -------
    None.

    '''
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 *
          recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 *
          np.mean(aurocs), 100 * np.std(aurocs)))
    print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 *
          np.mean(auprs), 100 * np.std(auprs)))

def get_error_rate(right_score, wrong_score, to_string=True):
    '''
    This function calculates the error rate of the model's prediction given the arrays of right
    and wrong scores.

    Parameters
    ----------
    right_score : array of float
        An array of confidence scores of the model's correct prediction.
    wrong_score : array of float
        An array of confidence scores of the model's incorrect prediction.
    to_string : bool, optional
        If true, print out the error rate of the model's prediction. The default is True.

    Returns
    -------
    error_rate : float
        The error rate of the model's prediction in percentage.
        
    '''
    num_right = len(right_score)
    num_wrong = len(wrong_score)

    error_rate = 100 * num_wrong / (num_wrong + num_right)

    if to_string:
        print('Error Rate {:.2f}'.format(error_rate))

    return error_rate

def get_id_measures(net, test_loader, device, temp='optimal', init_temp=1.5, score_function='MSP',
                    recall_level=0.95, to_string=True, method_name=''):
    '''
    This function calculates the performance measures obtained from an ID test dataset.

    Parameters
    ----------
    net : nn.Module
        The neural network model. 
    test_loader : torch.utils.data.DataLoader
        Data loader of the ID test dataset.
    device : str
        Device on which to perform confidence scores calculation.
    temp : float or str, optional
        The post-hoc temperature scale for confidence scores calculation. If 'optimal', tune 
        the tempearature scale using NLL optimization with the test dataset. The default is 
        'optimal'.
    score_function : str, optional
        The score function used to calculate the confidence scores. The default is 'MSP'.
    recall_level : float, optional
        The recall value or true positive rate (TPR) at which we calculate the false positive 
        rates (FPR). The default is 0.95. 
    to_string : str, optional
        If true, print out the calculated error rate, ECE error, and the performance measures of 
        AUROC, AUPR and FPR at a certain recall. The default is True.
    method_name : str, optional
        The method name of the model. The default is ''.

    Returns
    -------
    error_rate : float
        The error rate of the model's prediction in percentage.
    ece_error : float
        The expected calibration error (ECE) of the model given a test dataset.
    auroc : float
        The auroc score calculated from the test dataset, i.e. the area under the receiver operating 
        characteristic (AUROC).
    aupr : float
        The aupr score calculated from the test dataset, i.e. the area under the precision-recall 
        curve (AUPR). 
    fpr : float
        The fpr score at a certain recall calculated from the test dataset, i.e. the false positive 
        rate (FPR) at a certain recall.

    '''
    if temp == 'optimal':
        # optimize the post-hoc temperature scale with the ID test dataset
        temp = get_optimal_temperature(net, test_loader, device, init=init_temp)

    # /////////////// Detection Prelims ///////////////
    right_score, wrong_score = get_confidence_scores(net, test_loader, device, T=temp, score_function=score_function)

    # /////////////// Error Detection ///////////////
    error_rate = get_error_rate(right_score, wrong_score, to_string=to_string)
    # calculate the FPR95, AUROC, AUFR scores of the test result
    auroc, aupr, fpr = get_measures(-right_score, -wrong_score, recall_level=recall_level, 
                                        to_string=to_string, method_name=method_name)

    # /////////////// ECE Detection ///////////////
    ece_error = get_calibration_scores(net, test_loader, device, T=temp, n_bins=15, to_string=to_string)

    return error_rate, ece_error, auroc, aupr, fpr


def get_ood_measures(net, test_loader, ood_loader_dict, device, temp='optimal', init_temp=1.5, score_function='MSP',
                     recall_level=0.95, ood_num_examples=2000, test_bs=200, num_to_avg=10, to_string=True, 
                     method_name=''):
    '''
    This function calculates the performance measures obtained from an OOD test dataset.

    Parameters
    ----------
    net : nn.Module
        The neural network model. 
    test_loader : torch.utils.data.DataLoader
        Data loader of the ID test dataset.
    ood_loader_dict : dict of {str: torch.utils.data.DataLoader}
        A dictionary of {ood test data name, data loader of the corresponding OOD test dataset}.
    device : str
        Device on which to perform confidence scores calculation.
    temp : float or str, optional
        The post-hoc temperature scale for confidence scores calculation. If 'optimal', tune 
        the tempearature scale using NLL optimization with the test dataset. The default is 
        'optimal'.
    score_function : str, optional
        The score function used to calculate the confidence scores. The default is 'MSP'.
    recall_level : float, optional
        The recall value or true positive rate (TPR) at which we calculate the false positive 
        rates (FPR). The default is 0.95. 
    ood_num_examples : int, optional
        The number of examples extracted from the OOD test dataset for each test. The default is 
        2000.
    test_bs : int, optional
        The batch size of input to perform the test. The default is 200.
    num_to_avg : int, optional
        The number of trials to perform the test to get the average values of the performance 
        measures. The default is 10.
    to_string : str, optional
        If true, print out the mean values of the calculated performance measures, i.e. AUROC, AUPR 
        and FPR at a certain recall. The default is True.
    method_name : str, optional
        The method name of the model. The default is ''.

    Returns
    -------
    auroc_list : array of float
        THe array of auroc scores calculated from the test dataset, i.e. the area under the receiver operating 
        characteristic (AUROC).
    aupr_list : array of float
        The array of aupr scores calculated from the test dataset, i.e. the area under the precision-recall 
        curve (AUPR). 
    fpr_list : array of float
        The array of fpr scores at a certain recall calculated from the test dataset, i.e. the false positive 
        rate (FPR) at a certain recall.

    '''
    auroc_list, aupr_list, fpr_list = [], [], []
    
    if temp == 'optimal':
        # optimize the post-hoc temperature scale with the ID test dataset
        temp = get_optimal_temperature(net, test_loader, device, init=init_temp)

    # calculate the confidence scores of each ID data prediction
    in_score = get_ood_scores(net, test_loader, device, T=temp, score_function=score_function)
                                          
    for (data_name, ood_loader) in ood_loader_dict.items():

        if temp == 'optimal':
            # optimize the post-hoc temperature scale T with the OOD test dataset
            temp = get_optimal_temperature(net, ood_loader, device, init=init_temp)
        
        if to_string:
            print('\n\n{} Detection'.format(data_name))

        aurocs, auprs, fprs = [], [], []
        for _ in range(num_to_avg):
            # calculate the confidence scores of each OOD data prediction
            out_score = get_ood_scores(net, ood_loader, device, T=temp, score_function=score_function, 
                                       ood_num_examples=ood_num_examples, test_bs=test_bs)   
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
                print_measures_with_std(aurocs, auprs, fprs, recall_level=recall_level, method_name=method_name)
            else:
                print_measures(auroc, aupr, fpr, recall_level=recall_level, method_name=method_name)

        auroc_list.append(auroc)
        aupr_list.append(aupr)
        fpr_list.append(fpr)

    if to_string:
        # print mean results
        print('\n\nMean Test Results')
        print_measures(np.mean(auroc_list), np.mean(aupr_list),
                    np.mean(fpr_list), recall_level=recall_level, method_name=method_name)
    
    return auroc_list, aupr_list, fpr_list

def get_all_measures(net, test_loader, ood_loader_dict, device, temp='optimal', init_temp=1.5, score_function='MSP', 
                     recall_level=0.95, ood_num_examples=2000, test_bs=200, num_to_avg=10, to_string=True, 
                     method_name=''):
    '''
    This function calculates the performance measures obtained from an ID test dataset.

    Parameters
    ----------
    net : nn.Module
        The neural network model. 
    test_loader : torch.utils.data.DataLoader
        Data loader of the test dataset.
    ood_loader_dict : dict of {str: torch.utils.data.DataLoader}
        A dictionary of {ood test data name, data loader of the corresponding OOD test dataset}.
    device : str
        Device on which to perform confidence scores calculation.
    temp : float or str, optional
        The post-hoc temperature scale for confidence scores calculation. If 'optimal', tune 
        the tempearature scale using NLL optimization with the test dataset. The default is 
        'optimal'.
    score_function : str, optional
        The score function used to calculate the confidence scores. The default is 'MSP'.
    recall_level : float, optional
        The recall value or true positive rate (TPR) at which we calculate the false positive 
        rates (FPR). The default is 0.95. 
    ood_num_examples : int, optional
        The number of examples extracted from the OOD test dataset for each test. The default is 
        2000.
    test_bs : int, optional
        The batch size of input to perform the test. The default is 200.
    to_string : str, optional
        If true, print out the calculated ECE error. The default is True.
    method_name : str, optional
        The method name of the model. The default is ''.

    Returns
    -------
    error_rate : float
        The error rate of the model's prediction in percentage given a ID test dataset.
    ece_error : float
        The expected calibration error (ECE) of the model given a ID test dataset.
    auroc : float
        The auroc score calculated from the ID test dataset, i.e. the area under the receiver operating 
        characteristic (AUROC).
    aupr : float
        The aupr score calculated from the ID test dataset, i.e. the area under the precision-recall 
        curve (AUPR). 
    fpr : float
        The fpr score at a certain recall calculated from the ID test dataset, i.e. the false positive 
        rate (FPR) at a certain recall.
    auroc_list : array of float
        THe array of auroc scores calculated from the OOD test dataset, i.e. the area under the receiver 
        operating characteristic (AUROC).
    aupr_list : array of float
        The array of aupr scores calculated from the OOD test dataset, i.e. the area under the precision-
        recall curve (AUPR). 
    fpr_list : array of float
        The array of fpr scores at a certain recall calculated from the OOD test dataset, i.e. the false 
        positive rate (FPR) at a certain recall.

    '''
    error_rate, ece_error, auroc, aupr, fpr = get_id_measures(net, test_loader, device, temp=temp, init_temp=init_temp,
                                                              score_function=score_function, recall_level=recall_level, 
                                                              to_string=to_string, method_name=method_name)

    auroc_list, aupr_list, fpr_list = get_ood_measures(net, test_loader, ood_loader_dict, device, temp=temp, init_temp=init_temp,
                                                       score_function=score_function, recall_level=recall_level, 
                                                       ood_num_examples=ood_num_examples, test_bs=test_bs, num_to_avg=num_to_avg,
                                                       to_string=to_string, method_name=method_name)
    
    return error_rate, ece_error, auroc, aupr, fpr, auroc_list, aupr_list, fpr_list