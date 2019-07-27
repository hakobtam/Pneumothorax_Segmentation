import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from functools import partial
from torch.nn import functional as F

class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logit, target):
        output = torch.sigmoid(logit)

        intersection = torch.sum(target * output)
        union = torch.sum(target) + torch.sum(output) - intersection + self.eps
        return 1 - (intersection + self.eps) / union


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, beta=1.0):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, logit, target):
        output = torch.sigmoid(logit)

        tp = torch.sum(target * output)
        fp = torch.sum(output) - tp
        fn = torch.sum(target) - tp
        score = ((1 + self.beta ** 2) * tp + self.eps) \
                / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.eps)

        return 1 - score


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7):
        super().__init__(eps)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logit, target):
        jaccard = super().forward(logit, target)
        bce = self.bce(logit, target)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, beta=1.0, bce_weight=0.2):
        super().__init__(eps, beta)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_weight = bce_weight

    def forward(self, logit, target):
        dice = super().forward(logit, target)
        bce = self.bce(logit, target)
        return dice + self.bce_weight * bce

class Loss:
    def __init__(self, dice_weight=1):
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        assert ((outputs < 0) | (outputs > 1.0)).sum() == 0
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            smooth = 1e-8
            eps = 1e-7
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps

            loss -= self.dice_weight * torch.log(2 * (intersection+smooth) / (union+smooth))
            
        return loss

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1., eps=1e-7):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.eps = eps

#     def forward(self, output, target):
#         return 1 - (2 * torch.sum(output * target) + self.smooth) / (
#                 torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


# def mixed_dice_bce_loss(output, target, dice_weight=0.2, dice_loss=None,
#                         bce_weight=0.9, bce_loss=None,
#                         smooth=0, dice_activation='sigmoid'):

#     num_classes = output.size(1)
#     target = target[:, :num_classes, :, :].long()
#     if bce_loss is None:
#         bce_loss = nn.BCEWithLogitsLoss()
#     if dice_loss is None:
#         dice_loss = multiclass_dice_loss
#     return dice_weight * dice_loss(output, target, smooth, dice_activation) + bce_weight * bce_loss(output, target)


# Code has been taken from https://github.com/Hsuxu/Loss_ToolBox-PyTorch
class FocalLoss1(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class FocalLoss2(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

class FocalLoss3(nn.Module):
    """
    Arguments:
        gamma (float, optional): focusing parameter. Default: 2.
        alpha (float, optional): balancing parameter. Default: 0.25.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.
    """

    def __init__(self, gamma=2, alpha=4.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if reduction.lower() == "none":
            self.reduction_op = None
        elif reduction.lower() == "mean":
            self.reduction_op = torch.mean
        elif reduction.lower() == "sum":
            self.reduction_op = torch.sum
        else:
            raise ValueError(
                "expected one of ('none', 'mean', 'sum'), got {}".format(reduction)
            )

    def forward(self, input, target):
        if input.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(input.size(), target.size())
            )
        elif target.unique(sorted=True).tolist() not in [[0, 1], [0], [1]]:
            raise ValueError("target values are not binary")

        input = input.view(-1)
        target = target.view(-1)

        # Following the paper: probabilities = probabilities if y=1; otherwise,
        # probabilities = 1-probabilities
        probabilities = torch.sigmoid(input)
        probabilities = torch.where(target == 1, probabilities, 1 - probabilities)

        # Compute the loss
        focal = self.alpha * (1 - probabilities).pow(self.gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(input, target, reduction="none")
        loss = focal * bce

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss