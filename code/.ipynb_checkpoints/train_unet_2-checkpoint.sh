#!/usr/bin/bash

gpu=2
net=OUNet
num_epochs=101
batch_size=16
img_size=256
lr=0.0001
loss=ce_dice
python train1.py --Adam --net $net --num-epochs $num_epochs --batch-size 16 --img-size 256 --lr 0.0001 --loss $loss --gpu $gpu