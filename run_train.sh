#!/bin/bash

python train.py \
--model Unet \
--encoder resnet34 \
--pretrained imagenet \
--folds_dir 10folds \
--fold_id 0 \
--img_size 1024 \
--num_workers 4 \
--batch_size 8 \
--loss BCEDiceLoss \
--wd 1e-4 \
--optim adam \
--grad_accumulation 2 \
--lr 1e-4 \
--epochs 100 \
--seed 27 \
--resume ./runs/Aug15_19-47-17/checkpoints/Unet_best-dice_19.pt
#--resume ./runs/Aug03_13-03-43/checkpoints/Unet_best-dice_23.pt
#--resume-without-optimizer
#--freeze
#--debug
