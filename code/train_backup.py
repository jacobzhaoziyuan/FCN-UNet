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
import sys
import random
import argparse
import time
from utils import losses
from utils.logger import get_cur_time,checkpoint_save
from utils.lr import adjust_learning_rate,cosine_rampdown
import medpy.metric.binary as mmb
import pdb

basedir = '/home/ziyuan/UDA/code_crossmoda'

# train
data_path_train = '/diskd/ziyuan/crossmoda/process/npy/source_split/train_with_label.txt'
# val
data_path_val = '/diskd/ziyuan/crossmoda/process/npy/source_split/val.txt'


def get_arguments():

    parser = argparse.ArgumentParser(description='PyTorch UNet Training')


    # Model
    parser.add_argument('--num_classes', type=int, default=3,
                        help='output channel of network')
    parser.add_argument('--net', type=str, default='UNet', help='UNet/FCN/DRN')

    # Datasets
    parser.add_argument("--data-path-train", type=str, default=data_path_train,
                        help="Path to the images.")
    parser.add_argument("--data-path-val", type=str, default=data_path_val,
                        help="Path to the images.")
    parser.add_argument("--img-size", type=int, default=256)

    # Optimization options

    parser.add_argument('--batch-size', type=int,  default=8, help='batch size')
    parser.add_argument('--num-epochs', type=int,  default=50, help='maximum epoch number to train')
    parser.add_argument('--Adam', action='store_true')
    parser.add_argument('--SGD', action='store_true')
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument('--lr', type=float,  default=0.001, help='maximum epoch number to train')
    parser.add_argument("--lr-mode", default="constant", type=str, help='constant/consine')
    parser.add_argument("--change-optim", action="store_true")
    parser.add_argument('-Tmax', '--lr-rampdown-epochs', default=200, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--eta-min', default=0., type=float)

    # loss function options
    parser.add_argument('--loss', default='ce', type=str, help='ce/dice/ce_dice' )


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


def main():
    args.net = args.net.strip()
    args.loss = args.loss.strip()
    batch_size = args.batch_size
    base_lr = args.lr
    num_classes = args.num_classes


    logdir = os.path.join(basedir, 'logs', str(args.net), get_cur_time())
    print(logdir)
    savedir = os.path.join(basedir, 'checkpoints', str(args.net), get_cur_time())
    print(savedir)
    shotdir = os.path.join(basedir, 'snapshot', str(args.net), get_cur_time())
    print(shotdir)


    os.makedirs(logdir, exist_ok=False)
    os.makedirs(savedir, exist_ok=False)
    os.makedirs(shotdir, exist_ok=False)

    writer = SummaryWriter(logdir)


    logging.basicConfig(filename=shotdir+"/"+"snapshot.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))


    train_set = crossmoda_Dataset(args.data_path_train, phase = 'train', img_size = args.img_size)
    val_set = crossmoda_Dataset(args.data_path_val, phase = 'val')

    def worker_init_fn(worker_id):
        random.seed(args.manual_seed + worker_id)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    # print(args.net, args.net=='FCN')
    # pdb.set_trace()
    if args.net == 'UNet':
        # model = UNet(n_channels=1, n_classes=num_classes).cuda()  # old version
        model = UNet(in_chns=1, class_num=num_classes).cuda()
    elif args.net == 'FCN':
        model = FCN(num_cls=num_classes, pretrained=True, weights_init=None).cuda()
    elif args.net == 'DRN':
        model = DRN(num_cls=num_classes, weights_init=None).cuda()
    model.train()

    if args.SGD:
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              momentum=0.9, weight_decay=0.0001)
    if args.Adam:
        optimizer = optim.Adam(model.parameters(), lr=base_lr)

    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    performance = 0.0

    for epoch_num in tqdm(range(args.num_epochs), ncols=70):

        #TO-DO: SOME TRICKS FOR LR WARM UP.
        if args.warmup_epochs and epoch_num == args.warmup_epochs and args.change_optim:
            optimizer = optim.SGD(net.parameters(), lr=args.lr)
            print("changing to SGD")

        if epoch_num < args.warmup_epochs:
            lr = args.lr * (epoch_num + 1) / args.warmup_epochs  #gradual warmup_lr
            for param_group in optimizer.param_groups:
                print('lr --->', lr)
                param_group['lr'] = lr
        else:
            if isinstance(optimizer, optim.SGD) and args.lr_mode == "cosine":
                lr = adjust_learning_rate(optimizer, epoch_num - args.warmup_epochs, args.lr,
                                          args.lr_rampdown_epochs, args.eta_min) # args.lr
                for param_group in optimizer.param_groups:
                    print('lr --->', lr)
                    param_group['lr'] = lr


        loss_epoch = 0
        dice_epoch = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cuda()

            outputs = model(volume_batch)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice, _ = dice_loss(outputs, label_batch)

            if args.loss == 'ce':
                loss = loss_ce
            elif args.loss == 'dice':
                loss = loss_dice
            elif args.loss == 'ce_dice':
                loss = 0.5 * (loss_dice + loss_ce)

            dice_epoch += 1 - loss_dice.item()
            loss_epoch += loss.item()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

        epoch_dice = dice_epoch / len(trainloader)
        epoch_loss = loss_epoch / len(trainloader)
        writer.add_scalar('info/loss', epoch_loss, epoch_num +1)
        writer.add_scalar('info/dice', epoch_dice, epoch_num +1)

        logging.info('epoch %d : loss : %f dice: %f' % (epoch_num+1, epoch_loss, epoch_dice))

        if epoch_num % 10 == 0:
            # for validation
            model.eval()

            dice_list = []
            assd_list = []

            pred_vol_lst = [None for _ in range(20)]
            label_vol_lst = [None for _ in range(20)]

            for i_batch, sampled_batch in enumerate(valloader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                vol_id = sampled_batch['vol_id']-76
                volume_batch, label_batch = volume_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cpu().numpy()

                outputs = model(volume_batch)
                outputs = torch.softmax(outputs, dim = 1)
                outputs = torch.argmax(outputs, dim = 1).cpu().numpy()

                if pred_vol_lst[vol_id] is None:
                    pred_vol_lst[vol_id] = outputs
                else:
                    pred_vol_lst[vol_id] = np.vstack((pred_vol_lst[vol_id], outputs))
                if label_vol_lst[vol_id] is None:
                    label_vol_lst[vol_id] = label_batch
                else:
                    label_vol_lst[vol_id] = np.vstack((label_vol_lst[vol_id], label_batch))

            for i in range(20):
                for c in range(1, num_classes):
                    pred_test_data_tr = pred_vol_lst[i].copy()
                    pred_test_data_tr[pred_test_data_tr != c] = 0

                    pred_gt_data_tr = label_vol_lst[i].copy()
                    pred_gt_data_tr[pred_gt_data_tr != c] = 0

                    dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                    try:
                        assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))
                    except:
                        assd_list.append(np.nan)

            dice_arr = 100 * np.reshape(dice_list, [-1, 2]).transpose()
            dice_mean = np.mean(dice_arr, axis=1)

            assd_arr = np.reshape(assd_list, [-1, 2]).transpose()
            assd_mean = np.mean(assd_arr, axis=1)

            performance = dice_mean.mean()

            writer.add_scalar('info/val_assd', assd_mean.mean(), epoch_num + 1)
            writer.add_scalar('info/val_dice', dice_mean.mean(), epoch_num + 1)

            for class_i in range(1, num_classes):
                writer.add_scalar('info/val_{}_dice'.format(class_i),
                                  dice_mean[class_i-1], epoch_num + 1)
                writer.add_scalar('info/val_{}_assd'.format(class_i),
                                  assd_mean[class_i-1], epoch_num+1)

            # for i in 0, 1, 2 -> 1_dice, dice_mean[-1]

            logging.info('epoch %d : val_1_dice: %f val_1_dice: %f val_dice: %f val_assd : %f' % (epoch_num + 1,dice_mean[0], dice_mean[1], dice_mean.mean(), assd_mean.mean()))


        if performance > best_performance:
            checkpoint_save(model, performance > best_performance, savedir)
            logging.info("save model to {}".format(savedir))
            best_performance = max(performance, best_performance)

        model.train()

    if args.Adam:
        writer.add_hparams({'log_dir':logdir, 'arch/model':args.net, 'loss_func': args.loss,'optimizer': 'Adam', 'lr': args.lr, 'batch_size': args.batch_size, 'img_size':args.img_size,  'num_epoch':args.num_epochs}, {'val_dice': best_performance })
    elif args.SGD:
        writer.add_hparams({'log_dir':logdir, 'arch/model':args.net, 'loss_func': args.loss,'optimizer': 'SGD', 'lr': args.lr, 'batch_size': args.batch_size, 'img_size':args.img_size,  'num_epoch':args.num_epochs}, {'val_dice': best_performance })
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    main()
