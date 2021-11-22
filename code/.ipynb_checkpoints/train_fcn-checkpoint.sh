#!/usr/bin/bash
gpu=3
net=UNet
num_epochs=10
batch_size=8
img_size=256
lr=0.0001
loss=ce_dice
python train_fcn.py --Adam --net $net --num-epochs $num_epochs --batch-size $batch_size --img-size $img_size --lr $lr --loss $loss --gpu $gpu