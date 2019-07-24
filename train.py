import argparse
import os
import sys
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import *
import torchvision.utils as vutils

from metrices import *
from loss.bce_losses import Loss, FocalLoss
import models
import utils# import set_seed, create_optimizer, choose_device, create_lr_scheduler
from data_process.data_utils import *
from data_process import SIIMDataset

parser = argparse.ArgumentParser(description='Pneumothorax training')
parser.add_argument("--folds_dir", type=str, default='10folds', help='dataset folds directory')
parser.add_argument("--fold_id", type=int, default=0, help='dataset fold id to use for training')
parser.add_argument("--img_size", type=int, default=512, help='input image size')
parser.add_argument("--num_workers", type=int, default=4, help='number of workers')
parser.add_argument("--model", type=str, default='UNet', help='NN model name')
parser.add_argument("--num_filters", type=int, default=64, help='NN model number of filters')
parser.add_argument("--batch_size", type=int, default=32, help='batch size')
parser.add_argument("--loss", type=str, default='Loss', help='loss function')
parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
parser.add_argument("--optim", type=float, default='adam', help='optimization algorithm')
parser.add_argument('--grad_accumulation', type=int, default=1,
                    help='accumulate gradients over number of batches')
parser.add_argument("--lr", type=float, default=1e-2, help='learning rate for optimization')
#parser.add_argument("--lr_scheduler", type=float, default='step', help='method to adjust learning rate')
parser.add_argument("--epochs", type=int, default=350, help='number of epochs')
parser.add_argument("--debug", action='store_true', help='write debug images')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
default_log_dir = os.path.join('runs', current_time)
parser.add_argument('--log_dir', type=str, default=default_log_dir, help='location to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=27, help='seed value for random state')
#parser.add_argument('--pretrained', choices=('imagenet', 'coco', 'oid'), help='dataset name for pretrained model')
args = parser.parse_args()

#RandomChoice
#RandomApply
train_transform = Compose([PrepareData(),
                           HWCtoCHW()])
valid_transform = Compose([PrepareData(),
                           HWCtoCHW()])

os.makedirs(args.log_dir, exist_ok=True)
with open(os.path.join(args.log_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

train_dataset = SIIMDataset(subset='train', transform=train_transform, img_size=args.img_size,
                            folds_dir=args.folds_dir, fold_id=args.fold_id)
valid_dataset = SIIMDataset(subset='valid', transform=valid_transform, img_size=args.img_size,
                            folds_dir=args.folds_dir, fold_id=args.fold_id)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.num_workers, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                              num_workers=args.num_workers)

utils.set_seed(args.seed)
model = getattr(models, args.model)(args.num_filters)
parameters = [p for p in model.parameters() if p.requires_grad]
print('Number of parameters', len(parameters))
if args.optim == 'adam':
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)

if args.loss == "Loss":
    loss_fn = Loss()
#loss_fn = torch.nn.BCELoss()
#lr_scheduler = create_lr_scheduler(optimizer, **vars(args))

start_epoch = 0
best_loss = 1e10
best_dice = 0
best_iou = 0
best_acc = 0
tr_global_step = 0
val_global_step = 0

print("logging into {}".format(args.log_dir))
tensorboard_dir = os.path.join(args.log_dir, 'tensorboard')
os.makedirs(tensorboard_dir, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_dir)
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
# models_dir = os.path.join(args.log_dir, 'models')
# os.makedirs(models_dir, exist_ok=True)

def train(epoch, loss_fn):
    global tr_global_step, best_loss, best_iou, best_dice, best_acc, start_epoch
    writer.add_scalar('train/learning_rate', utils.get_lr(optimizer), epoch)
    
    model.train()
    torch.set_grad_enabled(True)
    optimizer.zero_grad()

    running_loss, running_iou, running_dice, running_acc = 0.0, 0.0, 0.0, 0.0
    it, total = 0, 0

    #pbar_disable = False if epoch == start_epoch else None
    pbar = tqdm(train_dataloader, unit="images", unit_scale=train_dataloader.batch_size, desc='Train')
    for batch in pbar:
        inputs, targets = batch['input'], batch['target']
        print("input shape: {}, output shape: {}".format(inputs.shape, targets.shape))
        inputs = inputs.cuda()
        targets = targets.cuda()

        # forward
        logits = model(inputs)
        probs = torch.sigmoid(logits).squeeze(1)
        predictions = probs > 0.5

        # logits = logits.squeeze(1)
        # targets = targets.squeeze(1)
        loss = loss_fn(torch.nn.Sigmoid(logits), targets)

        # accumulate gradients
        if it == 0:
            optimizer.zero_grad()
        loss.backward()
        if it % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        # statistics
        it += 1
        tr_global_step += 1
        loss = loss.item()
        running_loss += (loss * targets.size(0))
        total += targets.size(0)

        # writer.add_scalar('train/loss', loss, global_step)
        inputs_numpy = inputs.cpu().numpy()
        targets_numpy = targets.cpu().numpy().squeeze(1)
        probs_numpy = probs.cpu().detach().numpy().squeeze(1)
        predictions_numpy = probs_numpy > 0.5  # predictions.cpu().numpy()

        running_iou += iou_score(targets_numpy, predictions_numpy).sum()
        running_dice += dice_score(targets_numpy, predictions_numpy).sum()
        running_acc += accuracy_score(targets_numpy, predictions_numpy).sum()
        
        # update the progress bar
        pbar.set_postfix({
            'loss': "{:.05f}".format(running_loss / total),
            'IoU': "{:.03f}".format(running_iou / total),
            'Dice': "{:.03f}".format(running_dice / total),
            'Acc': "{:.03f}".format(running_acc / total)
        })

    epoch_loss = running_loss / total
    epoch_iou = running_iou / total
    epoch_dice = running_dice / total
    epoch_acc = running_acc / total
    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/iou', epoch_iou, epoch)
    writer.add_scalar('train/dice', epoch_dice, epoch)
    writer.add_scalar('train/accuracy', epoch_acc, epoch)

    return epoch_loss, epoch_iou, epoch_dice, epoch_acc

def validation(epoch, loss_fn):
    global val_global_step, best_loss, best_iou, best_dice, best_acc, start_epoch

    model.eval()
    torch.set_grad_enabled(False)
    running_loss, running_iou, running_dice, running_acc = 0.0, 0.0, 0.0, 0.0
    best_batch_loss, best_batch_dice, best_batch_iou, best_batch_acc = best_loss, best_dice, best_iou, best_acc
    it, total = 0, 0

    #pbar_disable = False if epoch == start_epoch else None
    pbar = tqdm(valid_dataloader, unit="images", unit_scale=valid_dataloader.batch_size, desc='Valid')
    for batch in pbar:
        inputs, targets = batch['input'], batch['target']
        print("input shape: {}, output shape: {}".format(inputs.shape, targets.shape))
        inputs = inputs.cuda()
        targets = targets.cuda()

        # forward
        logits = model(inputs)
        probs = torch.sigmoid(logits).squeeze(1)
        predictions = probs > 0.5

        # logits = logits.squeeze(1)
        # targets = targets.squeeze(1)
        loss = loss_fn(torch.nn.Sigmoid(logits), targets)

        # statistics
        it += 1
        val_global_step += 1
        loss = loss.item()
        running_loss += (loss * targets.size(0))
        total += targets.size(0)

        inputs_numpy = inputs.cpu().numpy()
        targets_numpy = targets.cpu().numpy().squeeze(1)
        probs_numpy = probs.cpu().detach().numpy().squeeze(1)
        predictions_numpy = probs_numpy > 0.5  # predictions.cpu().numpy()

        iou_array = iou_score(targets_numpy, predictions_numpy)
        dice_array = dice_score(targets_numpy, predictions_numpy)
        acc_array = dice_score(targets_numpy, predictions_numpy)
        iou = iou_array.mean()
        dice = dice_array.mean()
        acc = acc_array.mean()        
        running_iou += iou_array.sum()
        running_dice += dice_array.sum()
        running_acc += acc_array.sum()
        
        visualize_output = False
        if best_batch_loss > loss:
            best_batch_loss = loss
            visualize_output = True
        if best_batch_dice < dice:
            best_batch_dice = dice
            visualize_output = True
        if visualize_output and args.debug:
            # sort samples by metric
            ind = np.argsort(dice_array)
            images = inputs.cpu()
            images = images[ind]
            probs = probs[ind].cpu()
            predictions = predictions[ind].cpu()
            targets = targets[ind].cpu()

            preds = torch.cat([probs] * 3, 1)
            mask = torch.cat([targets.unsqueeze(1)] * 3, 1)
            all = images.clone()
            all[:, 0] = torch.max(images[:, 0], predictions.float())
            all[:, 1] = torch.max(images[:, 1], targets)
            all = torch.cat((torch.cat((all, images), 3), torch.cat((preds, mask), 3)), 2)
            all_grid = vutils.make_grid(all, nrow=4, normalize=False, pad_value=1)
            writer.add_image('valid/img-mask-pred', all_grid, val_global_step)
        # update the progress bar
        pbar.set_postfix({
            'loss': "{:.05f}".format(running_loss / total),
            'IoU': "{:.03f}".format(running_iou / total),
            'Dice': "{:.03f}".format(running_dice / total),
            'Acc': "{:.03f}".format(running_acc / total)
        })

    epoch_loss = running_loss / total
    epoch_iou = running_iou / total
    epoch_dice = running_dice / total
    epoch_acc = running_acc / total
    writer.add_scalar('valid/loss', epoch_loss, epoch)
    writer.add_scalar('valid/iou', epoch_iou, epoch)
    writer.add_scalar('valid/dice', epoch_dice, epoch)
    writer.add_scalar('valid/accuracy', epoch_acc, epoch)
    return epoch_loss, epoch_iou, epoch_dice, epoch_acc

print("training {}...".format(args.model))
pbar_epoch = tqdm(np.arange(start_epoch, args.max_epochs))

for epoch in np.arange(start_epoch, args.max_epochs):

    tr_epoch_loss, tr_epoch_iou, tr_epoch_dice, tr_epoch_acc = train(epoch)
    val_epoch_loss, val_epoch_iou, val_epoch_dice, val_epoch_acc = validation(epoch)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': epoch_loss,
        'best_metric': epoch_loss,
        'best_accuracy': epoch_loss,
    }
    if val_epoch_loss < best_loss:
        print ("Model Loss improved!!! {} -> {}".format(best_loss, val_epoch_loss))
        best_loss = val_epoch_loss
        utils.save_checkpoint(dir=checkpoint_dir, model=args.model , tag='best-loss', epoch=epoch)
    if val_epoch_dice > best_dice:
        print ("Model Dice improved!!! {} -> {}".format(best_dice, val_epoch_dice))
        best_dice = val_epoch_dice
        utils.save_checkpoint(dir=checkpoint_dir, model=args.model, tag='best-dice', epoch=epoch)
    if val_epoch_iou > best_iou:
        print ("Model IoU improved!!! {} -> {}".format(best_iou, val_epoch_iou))
        best_iou = val_epoch_iou
        utils.save_checkpoint(dir=checkpoint_dir, model=args.model, tag='best-iou', epoch=epoch)
    if val_epoch_acc > best_acc:
        print ("Model IoU improved!!! {} -> {}".format(best_acc, val_epoch_acc))
        best_acc= val_epoch_acc
        utils.save_checkpoint(dir=checkpoint_dir, model=args.model, tag='best-acc', epoch=epoch)



    # pbar_epoch.set_postfix({'lr': '{:.02e}' % utils.get_lr(optimizer),
    #                         'train': '%.03f/%.03f/%.03f' .format(
    #                         tr_epoch_loss, tr_epoch_iou, tr_epoch_dice, tr_epoch_acc),
    #                         'val': '%.03f/%.03f/%.03f'.format(
    #                         val_epoch_loss, val_epoch_iou, val_epoch_dice, val_epoch_acc),
    #                         'best val': '%.03f/%.03f/%.03f'.format(best_loss, best_dice, best_iou, best_acc)},
    #                        refresh=False)

utils.save_checkpoint('last')
print("best valid loss: {:.05f}, best valid dice: {:.03f}, best valid iou: {:.03f}, best valid accuracy: {:.03f},".\
     format(best_loss, best_dice, best_iou, best_acc))

