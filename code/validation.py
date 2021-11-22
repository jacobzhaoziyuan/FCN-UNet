import os
import torch
from dataloaders.dataset_crossmoda import crossmoda_Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from networks.unet2d import UNet
from networks.unet import UNet, OUNet
from networks.fcn8s import VGG16_FCN8s as FCN
from networks.drn import drn26 as DRN
import numpy as np
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse
import sys
import random
import time
import re
from utils import losses
from utils.logger import get_cur_time,checkpoint_save
import medpy.metric.binary as mmb
import pdb
import time

# val
data_path ='/diskd/ziyuan/crossmoda/process/npy/source_split/val.txt'
path =  '/home/ziyuan/UDA/code_crossmoda/checkpoints/UNet/2021-11-12_00-39-48/checkpoint.pth'
def get_arguments():

    parser = argparse.ArgumentParser(description='PyTorch UNet Training')


    # Model
    parser.add_argument('--num_classes', type=int, default=3,
                        help='output channel of network')
    parser.add_argument('--net', type=str, default='UNet')
    parser.add_argument('--path', type=str, default=path, help='path for saved model weight')

    # Datasets
    parser.add_argument("--data-path", type=str, default=data_path,
                        help="Path to the images.")

    # Miscs
    parser.add_argument('--manual-seed', type=int, default=1111, help='random seed')
    parser.add_argument('--gpu', type=str, default='2', help='GPU to use')

    return parser.parse_args()

args = get_arguments()
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)

np.random.seed(args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)


if __name__ == '__main__':
    num_classes = args.num_classes


    val_set = crossmoda_Dataset(data_path, phase = 'val')
    valloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=1)
    
    if args.net == 'UNet':
        # model = UNet(n_channels=1, n_classes=num_classes).cuda()  # old version
        model = UNet(in_chns=1, class_num=num_classes).cuda()
    if args.net == 'OUNet':
        model = OUNet(in_chns=1, class_num=num_classes).cuda()
    elif args.net == 'FCN':
        model = FCN(num_cls=num_classes, pretrained=True, weights_init=None).cuda()
    elif args.net == 'DRN':
        model = DRN(num_cls=num_classes, weights_init=None).cuda()
        
    model = model.cuda()
    print('load model from', args.path)
    model.load_state_dict(torch.load(args.path))
    model.eval()
    # os.makedirs(f'./result/{args.net}',exist_ok = True)
    
    
    dice_list = []
    assd_list = []

    
    vol_shape = np.load('/diskd/ziyuan/crossmoda/process/npy/source_split/vol_shape.npy')
    pred_vol_lst = [np.zeros((i,256,256)) for i in vol_shape[76:96] ]
    label_vol_lst = [np.zeros((i,256,256)) for i in vol_shape[76:96] ]

    # time_start = time.time
    for i_batch, sampled_batch in enumerate(tqdm(valloader)):
        # pdb.set_trace()
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        vol_id = sampled_batch['vol_id']-76
        slice_id = sampled_batch['slice_id']
        volume_batch, label_batch = volume_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cpu().numpy()

        outputs = model(volume_batch)
        outputs = torch.softmax(outputs, dim = 1)
        outputs = torch.argmax(outputs, dim = 1).cpu().numpy()
        # if pred_vol_lst[vol_id] is not None:
        #     print(outputs.shape, pred_vol_lst[vol_id].shape)
        
        # if pred_vol_lst[vol_id] is None:
        #     pred_vol_lst[vol_id] = outputs
        # else:
        #     pred_vol_lst[vol_id] = np.vstack((pred_vol_lst[vol_id], outputs))
        # if label_vol_lst[vol_id] is None:
        #     label_vol_lst[vol_id] = label_batch
        # else:
        #     label_vol_lst[vol_id] = np.vstack((label_vol_lst[vol_id], label_batch))
        
        pred_vol_lst[vol_id][slice_id] = outputs[0,...]
        label_vol_lst[vol_id][slice_id] = label_batch[0,...]
        
    # time_end = time.time

    for i in range(20):
        for c in range(1, num_classes):   
            pred_test_data_tr = pred_vol_lst[i].copy()              
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label_vol_lst[i].copy() 
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
            assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))
    # pdb.set_trace()
    dice_arr = 100 * np.reshape(dice_list, [-1, 2]).transpose()
    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    assd_arr = np.reshape(assd_list, [-1, 2]).transpose()
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std = np.std(assd_arr, axis=1)
    
    # with open(f'./result/{args.net}/val_res.txt','w') as f:
    with open(f'./result/optim/val_res.txt','w') as f:
        print(dice_mean, assd_mean, file=f)
        print('Dice:',file=f)
        print('VS :%.1f(%.1f)' % (dice_mean[0], dice_std[0]),file=f)
        print('cochlea:%.1f(%.1f)' % (dice_mean[1], dice_std[1]),file=f)
        print('Mean:%.1f' % np.mean(dice_mean),file=f)

        print('ASSD:',file=f)
        print('VS :%.1f(%.1f)' % (assd_mean[0], assd_std[0]),file=f)
        print('cochlea:%.1f(%.1f)' % (assd_mean[1], assd_std[1]),file=f)
        print('Mean:%.1f' % np.mean(assd_mean),file=f)

        # print('Inference time: %.1f'% (time_start-time_end)/10,file=f)
