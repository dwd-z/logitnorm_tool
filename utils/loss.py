import torch
import torch.nn as nn
import torch.nn.functional as F

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
            The raw, unnormalized output logits of the neural network for each class.  
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

    def __init__(self, n_bins=15):
        '''
        This method initiates a _ECELoss object.

        Parameters
        ----------
        n_bins : int
            number of confidence interval bins

        '''
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t=1.0):
        '''
        This forward function calculates the Expected Calibration Error (ECE) of a model. This 
        divides the confidence scores into equally-sized interval bins.
        
        In each bin, we compute the confidence gap:
        bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
        We then return a weighted average of the gaps, based on the number of samples in each bin.
        See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht. "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
        2015.

        Parameters
        ----------
        logits : array of float
            The raw output of the model before softmax score function.
        labels : array of int
            The ground truth, i.e. true labels of the input dataset.
        t : float, optional
            The temperature scale for confidence scores calculation. The default is 1.0, 
      
        Returns
        -------
        ece : 
            The Expected Calibration Error (ECE) of the model calculated from the logits.

        '''
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