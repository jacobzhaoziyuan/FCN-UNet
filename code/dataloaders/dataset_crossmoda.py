from torch.utils.data import Dataset
import os
import numpy as np
import re
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms
import cv2
import random
random.seed(1)


class crossmoda_Dataset(Dataset):

    def __init__(self, data_path, phase = 'train', img_size = 256):

        with open(data_path, 'r') as f:
            img_list = f.readlines()
        self.img_list = [img.strip() for img in img_list]
        
        if phase == 'train':
            self.transform = True
        else:
            self.transform = False
        self.img_size = img_size

        print("total {} samples".format(len(self.img_list)))
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = re.sub('\.npy','_label.npy', img_path)
        volume_id = int(re.findall('crossmoda_([0-9]+)_.+',img_path)[0])
        slice_id = int(re.findall('crossmoda_.+_([0-9]+)',img_path)[0])
        
        img = np.load(img_path)
        label = np.load(label_path).astype(np.uint8)

        if self.transform:
            seq = iaa.Sequential([iaa.Resize(size = self.img_size), iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10))])
            segmap = SegmentationMapsOnImage(label, shape=label.shape)
            img_aug, segmaps_aug = seq(image=img, segmentation_maps=segmap)
            label_aug = segmaps_aug.arr
            return  {'image': img_aug[np.newaxis,...], 'label': label_aug[:,:,0].astype(np.uint8), 'vol_id':volume_id, 'slice_id':slice_id} 
        else:
            return  {'image': img[np.newaxis,...], 'label': label.astype(np.uint8), 'vol_id':volume_id, 'slice_id':slice_id} 

if __name__ =="__main__":
    # train
    data_path = '/diskd/ziyuan/crossmoda/process/npy/source_split/train_with_label.txt'
    # val
    # data_path = '/diskd/ziyuan/crossmoda/process/npy/source_split/val.txt'
    # test
    # data_path = '/diskd/ziyuan/crossmoda/process/npy/source_split/test.txt'
    train_set = crossmoda_Dataset(data_path, phase = 'train', img_size=512)
    sample = train_set[0]
    print(sample['image'].shape, sample['label'].shape,sample['vol_id'], sample['slice_id'])
    # print(sample['image'].shape, sample['imgname'])
