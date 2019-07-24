#!/usr/bin/env bash

python train.py \
--model UNet \
--num_filters 32 \
--folds_dir 10folds \
--fold_id 0 \
--img_size 512 \
--num_workers 4 \
--batch_size 16 \
--loss Loss \
--wd 1e-4 \
--optim adam \
--grad_accumulation 1 \
--lr 1e-2 \
#--lr_scheduler step \
--epochs 100 \
--seed 27 \
--debug
