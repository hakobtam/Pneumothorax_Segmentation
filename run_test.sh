#!/bin/bash

python test.py \
--num_workers 4 \
--batch_size 16 \
--seed 27 \
--threshold 0.5 \
--ckpt ./runs/Aug17_21-35-25/checkpoints/Unet_best-dice_39.pt \
--tta \
--use_gpu