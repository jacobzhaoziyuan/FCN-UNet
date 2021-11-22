import tensorflow as tf
import numpy as np
import os.path as osp
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt

def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:

        features = tf.train.Features(
            feature = {
                "data_vol": tf.train.Feature(bytes_list = tf.train.BytesList(value = [data.astype(np.float32).tostring()])),
                "label_vol": tf.train.Feature(bytes_list = tf.train.BytesList(value = [label.astype(np.float32).tostring()])),
                'dsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value= [np.int64(data.shape[0])])),
                'dsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value= [np.int64(data.shape[1])])),
                'dsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value= [np.int64(data.shape[2])])),
                'lsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value= [np.int64(label.shape[0])])),
                'lsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value= [np.int64(label.shape[1])])),
                'lsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value= [np.int64(label.shape[2])])),
            }
        )
        example = tf.train.Example(features = features)
        serialized = example.SerializeToString()
        writer.write(serialized)
        
seq = iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10))
# seq_det = seq.to_deterministic()

data_root = '/diskd/ziyuan/crossmoda/process/npz'
phases = ['source_training', 'target_training', 'target_val']
tf_savedir = '/diskd/ziyuan/crossmoda/process/tfrecords'

for p in phases:
    if not os.path.exists(osp.join(tf_savedir,p)):
        os.makedirs(osp.join(tf_savedir,p))
        
        
path = '/diskd/ziyuan/crossmoda/process/npz/source_training'
save_path = '/diskd/ziyuan/crossmoda/process/tfrecords/source_training'
aug_num = 5
for n in range(1,2):
    npz = osp.join(path,'crossmoda_'+str(i)+'.npz')
    A = np.load(npz)
    data = A['arr_0'] # (*,256,256)
    label = A['arr_1'] # (*,256,256)
    data = np.concatenate([i[..., None] for i in data], axis=-1) # (256,256,*)
    label = np.concatenate([i[..., None] for i in label], axis=-1) # (256,256,*)
    print(n,data.shape)
    for j in range(1, data.shape[2]-2):
        # get three consecutive slice
        label_slices = label[:,:,j-1:j+2]
        data_slices = data[:,:,j-1:j+2]
        save_tfrecords(data_slices, label_slices, os.path.join(save_path,str(n) +'_'+str(j)+ "_0.tfrecords"))
        segmap = SegmentationMapsOnImage(label_slices, shape=label_slices.shape)
        for k in range(1,aug_num):
            img_aug, segmaps_aug = seq(image=data_slices, segmentation_maps=segmap)
            save_tfrecords(img_aug, segmaps_aug.arr, os.path.join(save_path,str(n) +'_'+str(j)+ "_"+str(k)+".tfrecords"))