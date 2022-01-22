FCN & UNet Implementation: Vestibular Schwannoma and Cochlea Segmentation from Contrast-enhanced T1 MRI



## Abstract
Segmentation is one of the most important steps in medical image analysis. It aims to accurately divide the image into meaningful groups. Recently, deep convolutional neural networks, particularly fully convolution networks, have achieved state-of-the-art results on semantic image segmentation.  In this work, we mainly reviewed two representative fully convolution networks, i.e., FCN and UNet, and implemented them in one challenging segmentation task, vestibular schwannoma and cochlea segmentation. 

## Reimplemented methods
* [FCN](https://arxiv.org/abs/1411.4038), [UNet](https://arxiv.org/abs/1505.04597), and [DRN](https://arxiv.org/abs/1705.09914)




## Setup

1. Follow official guidance to install [Pytorch](https://pytorch.org/).
2. Clone the repo
3. cd code


## Data Preparation
Cross-Modality Domain Adaptation for Medical Image Segmentation Challenge (CrossMoDA) dataset
https://crossmoda-challenge.ml/

Only source training dataset (contrast-enhanced T1) was used.

Please use `utils/crossmoda/preprocess.ipynb` to explore the data preprocessing process.
    

## Training

Run `bash train.sh`


## Evaluation 
Run `validation.py` 


## Visualization
Please use `utils/visualization.ipynb`





## Citation
If you find the codebase useful for your research, please cite the papers:
```

@inproceedings{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3431--3440},
  year={2015}
}

@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}

@inproceedings{yu2017dilated,
  title={Dilated residual networks},
  author={Yu, Fisher and Koltun, Vladlen and Funkhouser, Thomas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={472--480},
  year={2017}
}

@inproceedings{zhao2021mt,
  title={MT-UDA: Towards Unsupervised Cross-modality Medical Image Segmentation with Limited Source Labels},
  author={Zhao, Ziyuan and Xu, Kaixin and Li, Shumeng and Zeng, Zeng and Guan, Cuntai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={293--303},
  year={2021},
  organization={Springer}
}
```

## Acknowledgement

Part of the code is adapted from open-source codebase and original implementations of algorithms, 
we thank these authors for their fantastic and efficient codebase:
* UNet: https://github.com/zhixuhao/unet
* FCN & DRN: https://github.com/jhoffman/cycada_release/tree/master/cycada/models
