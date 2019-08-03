import numpy as np

def dice_score(y_true, y_pred, noise_th, thresh=0.5):
    epsilon = 1e-15
    y_true = y_true > 0.5
    y_pred = y_pred > thresh
    y_pred[y_pred.sum((1,2)) < noise_th] = 0.0 
    intersection = (y_true & y_pred).sum((1, 2))
    union = y_true.sum((1, 2)) + y_pred.sum((1, 2))
    dice = 2 * intersection / (union + epsilon)
    dice[union == 0] = 1.
    return dice

def iou_score(y_true, y_pred, thresh=0.5):
    epsilon = 1e-15
    y_true = y_true > 0.5
    y_pred = y_pred > thresh
    intersection = (y_true & y_pred).sum((1, 2))
    union = (y_true | y_pred).sum((1, 2))
    iou = intersection / (union + epsilon)
    return iou

def accuracy_score(y_true, y_pred, thresh=0.5):
    y_true = y_true > 0.5
    y_pred = y_pred > thresh
    acc = (y_true == y_pred).sum((1,2)) / y_true[0, :, :].size
    return acc