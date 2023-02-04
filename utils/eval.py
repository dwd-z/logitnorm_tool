from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sklearn.metrics as metrics

from utils.loss import _ECELoss
from utils.posthoc import odin

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