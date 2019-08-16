#!/bin/bash

python test.py \
--num_workers 4 \
--batch_size 16 \
--seed 27 \
--threshold 0.5 \
--ckpt ./runs/Aug15_22-29-05/checkpoints/Unet_best-dice_22.pt \
--tta \
--use_gpu