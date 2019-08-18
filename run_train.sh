#!/bin/bash

python train.py \
--model Unet \
--encoder resnet34 \
--pretrained imagenet \
--folds_dir 10folds \
--fold_id 0 \
--img_size 512 \
--num_workers 4 \
--batch_size 16 \
--loss criterion \
--wd 1e-3 \
--optim adam \
--grad_accumulation 2 \
--lr 1e-4 \
--epochs 100 \
--seed 27 \
#--resume Pneumothorax_Segmentation/runs/Aug15_22-29-05/checkpoints/Unet_best-dice_22.pt
#--resume ./runs/Aug15_19-47-17/checkpoints/Unet_best-dice_19.pt
#--resume-without-optimizer
#--freeze
#--debug
