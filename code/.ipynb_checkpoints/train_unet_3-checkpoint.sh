#!/usr/bin/bash

gpu=3
net=FCN
num_epochs=101
batch_size=16
img_size=256
lr=0.0001
loss=ce_dice
python train.py --Adam --net UNet --num-epochs 201 --batch-size 64 --img-size $img_size --lr 0.0001 --loss $loss --gpu $gpu &&
python train.py --Adam --net $net --num-epochs $num_epochs --batch-size 16 --img-size $img_size --lr 0.0001 --loss $loss --gpu $gpu &&
python train.py --Adam --net $net --num-epochs $num_epochs --batch-size 16 --img-size $img_size --lr 0.00001 --loss $loss --gpu $gpu &&
python train.py --Adam --net $net --num-epochs $num_epochs --batch-size 16 --img-size $img_size --lr 0.001 --loss $loss --gpu $gpu &&
python train.py --Adam --net $net --num-epochs $num_epochs --batch-size 8 --img-size $img_size --lr 0.00001 --loss $loss --gpu $gpu &&
python train.py --Adam --net $net --num-epochs $num_epochs --batch-size 8 --img-size $img_size --lr 0.0001 --loss $loss --gpu $gpu &&
python train.py --Adam --net $net --num-epochs $num_epochs --batch-size 8 --img-size $img_size --lr 0.001 --loss $loss --gpu $gpu &&
python train.py --SGD --net $net --num-epochs $num_epochs --batch-size 16 --img-size $img_size --lr 0.0001 --loss $loss --gpu $gpu