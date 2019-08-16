import argparse
import os
import sys
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

from tensorboardX import SummaryWriter
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import *
import torchvision.utils as vutils

from metrices import *
from loss.bce_losses import *
from loss.lovasz_losses import lovasz_hinge, binary_xloss
import models
import utils# import set_seed, create_optimizer, choose_device, create_lr_scheduler
from data_process.data_utils import *
from data_process import SIIMDataset
from efficientnet_pytorch import EfficientNet


from pdb import set_trace

parser = argparse.ArgumentParser(description='Pneumothorax training')
parser.add_argument("--folds_dir", type=str, default='10folds', help='dataset folds directory')
parser.add_argument("--fold_id", type=int, default=0, help='dataset fold id to use for training')
parser.add_argument("--img_size", type=int, default=512, help='input image size')
parser.add_argument("--num_workers", type=int, default=4, help='number of workers')
parser.add_argument("--model", type=str, default='Unet', help='NN model name')
parser.add_argument("--encoder", type=str, default='resnet50', help='encoder name')
#parser.add_argument("--num_filters", type=int, default=64, help='NN model number of filters')
parser.add_argument("--batch_size", type=int, default=32, help='batch size')
parser.add_argument("--loss", type=str, default='Loss', help='loss function')
parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
parser.add_argument("--optim", type=str, default='adam', help='optimization algorithm')
parser.add_argument('--grad_accumulation', type=int, default=1,
                    help='accumulate gradients over number of batches')
parser.add_argument("--lr", type=float, default=1e-2, help='learning rate for optimization')
#parser.add_argument("--lr_scheduler", type=float, default='step', help='method to adjust learning rate')
parser.add_argument("--epochs", type=int, default=350, help='number of epochs')
parser.add_argument("--debug", action='store_true', help='write debug images')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument('--resume-without-optimizer', action='store_true', help='resume but don\'t use optimizer state')
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
default_log_dir = os.path.join('runs', current_time)
parser.add_argument('--log_dir', type=str, default=default_log_dir, help='location to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=27, help='seed value for random state')
parser.add_argument("--freeze", action='store_true', help='freeze encoder weights')
parser.add_argument('--pretrained', default='imagenet', choices=('imagenet', 'coco', 'oid'), help='dataset name for pretrained model')
args = parser.parse_args()

#RandomChoice
#RandomApply
utils.set_seed(args.seed)
#torch.backends.cudnn.benchmark = True
if args.model == "EfficientNet":
    model_name = 'efficientnet-b6'
    image_size = EfficientNet.get_image_size(model_name) 
    model = EfficientNet.from_pretrained(model_name, num_classes=1)
    preprocessing_transform = Compose([Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
else:
    preprocessing_fn = models.encoders.get_preprocessing_fn(args.encoder, args.pretrained)
    preprocessing_transform = Compose([preprocessing_fn])
    model = getattr(models, args.model)(
                                        encoder_name=args.encoder,
                                        encoder_weights=args.pretrained, 
                                        classes=1, 
                                        activation='sigmoid'
                                    )
model.cuda()

train_transform = Compose([PrepareData()])
valid_transform = Compose([PrepareData()])

#print(sys.argv)
os.makedirs(args.log_dir, exist_ok=True)
with open(os.path.join(args.log_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

tr_data_len = None
val_data_len = None
if args.debug:
    tr_data_len = 1000
    val_data_len = 100
train_dataset = SIIMDataset(subset='train', transform=train_transform, preprocessing=preprocessing_transform, 
                            img_size=args.img_size, folds_dir=args.folds_dir, fold_id=args.fold_id, data_len=tr_data_len)
valid_dataset = SIIMDataset(subset='valid', transform=valid_transform, preprocessing=preprocessing_transform, 
                            img_size=args.img_size, folds_dir=args.folds_dir, fold_id=args.fold_id, data_len=val_data_len)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.num_workers, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                              num_workers=args.num_workers, drop_last=True)


if args.freeze:
    for param in model.encoder.parameters():
        param.requires_grad = False

# for name, param in model.encoder.named_parameters():
#     print(name)
# exit()
#parameters = [p for p in model.parameters() if p.requires_grad]
parameters = [
                {'params': [p for name, p in model.encoder.named_parameters() if 'layer1' in name], 'lr': args.lr/10000},
                {'params': [p for name, p in model.encoder.named_parameters() if 'layer2' in name], 'lr': args.lr/1000},
                {'params': [p for name, p in model.encoder.named_parameters() if 'layer3' in name], 'lr': args.lr/100},
                {'params': [p for name, p in model.encoder.named_parameters() if 'layer4' in name], 'lr': args.lr/10},
                {'params': [p for name, p in model.decoder.named_parameters()]}
            ]
print('Number of parameters', len(parameters))
if args.optim == 'adam':
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)

if args.loss == "Loss":
    loss_fn = Loss(0.1)
if args.loss == "FocalLoss":
    loss_fn = FocalLoss2()
if args.loss == "BCEDiceLoss":
    loss_fn = BCEDiceLoss(bce_weight=0.2)
if args.loss == "lovasz":
    loss_fn = lovasz_hinge
if args.loss == "criterion":
    loss_fn = criterion
# if args.loss == "DiceLoss":
#     loss_fn = mixed_dice_bce_loss()
#loss_fn = torch.nn.BCELoss()
#lr_scheduler = create_lr_scheduler(optimizer, **vars(args))

start_epoch = 0
best_loss = 1e10
best_dice = 0
best_iou = 0
best_acc = 0
tr_global_step = 0
val_global_step = 0
noise_th=75.0*(args.img_size/128.0)**2

if args.resume:
    print("resuming a checkpoint '%s'" % args.resume)
    if os.path.exists(args.resume):
        saved_checkpoint = torch.load(args.resume)
        model.load_state_dict(saved_checkpoint['model_state'])

        if not args.resume_without_optimizer:
            optimizer.load_state_dict(saved_checkpoint['optimizer_state'])
            #lr_scheduler.load_state_dict(saved_checkpoint['lr_scheduler'])
            best_loss = saved_checkpoint.get('loss_val', best_loss)
            best_dice = saved_checkpoint.get('dice_val', best_dice)
            best_iou = saved_checkpoint.get('iou_val', best_iou)
            best_acc = saved_checkpoint.get('acc_val', best_acc)
            start_epoch = saved_checkpoint.get('epoch', start_epoch)
            #global_step = saved_checkpoint.get('step', global_step)

        del saved_checkpoint  # reduce memory
    else:
        print(">\n>\n>\n>\n>\n>")
        print(">Warning the checkpoint '%s' doesn't exist! training from scratch!" % args.resume)
        print(">\n>\n>\n>\n>\n>")

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
    pbar = tqdm(train_dataloader, unit="images", unit_scale=train_dataloader.batch_size, desc='Train: epoch {}'.format(epoch))
    for batch in pbar:
        inputs, targets = batch['input'], batch['target']
        #print("input shape: {}, output shape: {}".format(inputs.shape, targets.shape))
        inputs = inputs.float().cuda()
        targets = targets.cuda()

        # forward
        logits = model(inputs)
        logits = logits.squeeze(1)
        targets = targets.squeeze(1)

        probs = torch.sigmoid(logits)
        loss = loss_fn(logits, targets)
        # accumulate gradients
        loss = loss / args.grad_accumulation    
        loss.backward()
        if (it + 1) % args.grad_accumulation == 0:
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
        targets_numpy = targets.cpu().numpy()
        probs_numpy = probs.cpu().detach().numpy()
        predictions_numpy = probs_numpy > 0.5  # predictions.cpu().numpy()

        running_iou += iou_score(targets_numpy, predictions_numpy).sum()
        running_dice += dice_score(targets_numpy, predictions_numpy, noise_th=noise_th).sum()
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
    pbar = tqdm(valid_dataloader, unit="images", unit_scale=valid_dataloader.batch_size, desc='Valid: epoch {}'.format(epoch))
    for batch in pbar:
        inputs, inputs_img, targets = batch['input'], batch['input_img'], batch['target']
        #print("input shape: {}, output shape: {}".format(inputs.shape, targets.shape))
        inputs = inputs.float().cuda()
        targets = targets.cuda()

        # forward
        logits = model(inputs)
        logits = logits.squeeze(1)
        targets = targets.squeeze(1)

        probs = torch.sigmoid(logits)
        loss = loss_fn(logits, targets)

        # statistics
        it += 1
        val_global_step += 1
        loss = loss.item()
        running_loss += (loss * targets.size(0))
        total += targets.size(0)

        inputs_numpy = inputs.cpu().numpy()
        targets_numpy = targets.cpu().numpy()
        probs_numpy = probs.cpu().detach().numpy()
        predictions_numpy = probs_numpy > 0.5  # predictions.cpu().numpy()

        iou_array = iou_score(targets_numpy, predictions_numpy)
        dice_array = dice_score(targets_numpy, predictions_numpy, noise_th=noise_th)
        acc_array = accuracy_score(targets_numpy, predictions_numpy)
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
        if visualize_output:
            # sort samples by metric
            ind = np.argsort(dice_array)
            images = inputs_img[ind]
            probs = probs[ind].cpu()
            predictions = (probs > 0.5).float()
            targets = targets[ind].cpu()

            #images=torch.cat([images] * 3, 1)
            all1 = images.clone()
            all1[:, 0] = torch.max(images[:, 0], probs)
            all1[:, 2] = torch.max(images[:, 2], targets)
            all2 = images.clone()
            all2[:, 0] = torch.max(images[:, 0], predictions)
            all2[:, 2] = torch.max(images[:, 2], targets)
            masks = torch.zeros(images.shape)
            masks[:, 0] = predictions
            masks[:, 2] = targets
            all = torch.cat([images, all1, all2, masks], 3)
            all_grid = vutils.make_grid(all, nrow=1, normalize=False, pad_value=1)
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


if __name__ == '__main__':
    print("training {}...".format(args.model))
    #pbar_epoch = tqdm(np.arange(start_epoch, args.epochs))

    for epoch in np.arange(start_epoch, args.epochs):
        loss_fn_to_pass=loss_fn
        tr_epoch_loss, tr_epoch_iou, tr_epoch_dice, tr_epoch_acc = train(epoch, loss_fn_to_pass)
        val_epoch_loss, val_epoch_iou, val_epoch_dice, val_epoch_acc = validation(epoch, loss_fn_to_pass)
        state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_val': val_epoch_loss,
            'dice_val': val_epoch_dice,
            'iou_val': val_epoch_iou,
            'acc_val': val_epoch_acc,
        }
        state.update(vars(args))
        mark = 0
        if val_epoch_loss < best_loss:
            mark = 1
            print ("Model Loss improved!!! {} -> {}".format(best_loss, val_epoch_loss))
            best_loss = val_epoch_loss
            utils.save_checkpoint(ckpt_dir=checkpoint_dir, model=args.model , tag='best-loss', epoch=epoch, save_dict=state)
        if val_epoch_dice > best_dice:
            mark = 1
            print ("Model Dice improved!!! {} -> {}".format(best_dice, val_epoch_dice))
            best_dice = val_epoch_dice
            utils.save_checkpoint(ckpt_dir=checkpoint_dir, model=args.model, tag='best-dice', epoch=epoch, save_dict=state)
        if val_epoch_iou > best_iou:
            mark = 1
            print ("Model IoU improved!!! {} -> {}".format(best_iou, val_epoch_iou))
            best_iou = val_epoch_iou
            utils.save_checkpoint(ckpt_dir=checkpoint_dir, model=args.model, tag='best-iou', epoch=epoch, save_dict=state)
        if val_epoch_acc > best_acc:
            mark = 1
            print ("Model Acc improved!!! {} -> {}".format(best_acc, val_epoch_acc))
            best_acc= val_epoch_acc
            utils.save_checkpoint(ckpt_dir=checkpoint_dir, model=args.model, tag='best-acc', epoch=epoch, save_dict=state)
        if mark == 0:
            print("Model didn't improved")


        # pbar_epoch.set_postfix({'lr': '{:.02e}' % utils.get_lr(optimizer),
        #                         'train': '%.03f/%.03f/%.03f' .format(
        #                         tr_epoch_loss, tr_epoch_iou, tr_epoch_dice, tr_epoch_acc),
        #                         'val': '%.03f/%.03f/%.03f'.format(
        #                         val_epoch_loss, val_epoch_iou, val_epoch_dice, val_epoch_acc),
        #                         'best val': '%.03f/%.03f/%.03f'.format(best_loss, best_dice, best_iou, best_acc)},
        #                        refresh=False)

    #utils.save_checkpoint('last')
    print("best valid loss: {:.05f}, best valid dice: {:.03f}, best valid iou: {:.03f}, best valid accuracy: {:.03f},".\
        format(best_loss, best_dice, best_iou, best_acc))

