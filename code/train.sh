#!/usr/bin/bash

gpu=0
net=DRN
num_epochs=101
batch_size=16
img_size=256
lr=0.0001
loss=ce_dice
python train.py --Adam --net  $net --num-epochs 101 --batch-size 64 --img-size $img_size --lr 0.0001 --loss $loss --gpu $gpu