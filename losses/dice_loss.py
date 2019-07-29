import torch
import torch.nn as nn


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
