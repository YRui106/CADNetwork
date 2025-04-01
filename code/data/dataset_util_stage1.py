import os
import glob
import torch
from torch.utils.data import Dataset
from numpy.random import RandomState

import matplotlib.pyplot as plt
import numpy as np
import cv2
#corrupted_files = open('corrupted_fils', 'w')


class RainDataset(Dataset):
    def __init__(self, opt, is_eval=False, is_test=False):
        super(RainDataset, self).__init__()

        if is_test:
            self.dataset = opt.test_dataset
        elif is_eval:
            self.dataset = opt.eval_dataset
        else:
            self.dataset = opt.train_dataset
        # dataset = open(self.dataset, 'r').read().split()
        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
        self.depth_list = sorted(glob.glob(self.dataset+'/depth/*'))
        self.rand_state = RandomState(66)   #是不是相当于seed
        self.patch_size = opt.patch_size
   
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]
        depth_name = self.depth_list[idx]

        img = cv2.imread(img_name,-1)
        gt = cv2.imread(gt_name,-1)   #以彩色方式读入
        depth = cv2.imread(depth_name,-1)
        
        img , gt , depth = self.crop(img, gt, depth)  #crop
        if img.dtype == np.uint8:   #调用这里
            img = (img / 255.0).astype('float32')
#            print("归一化")
        if gt.dtype == np.uint8:
            gt = (gt / 255.0).astype('float32')
            
        if depth.dtype == np.uint8:
            depth = (depth / 255.0).astype('float32')

        return [img,gt,depth]

    def crop(self, img_pair,gt, depth):
        patch_size = self.patch_size
        h, w, c = img_pair.shape
        r = self.rand_state.randint(0, h - patch_size)
        c = self.rand_state.randint(0, w - patch_size)
        B = img_pair[r: r+patch_size, c: c+patch_size]
        GT = gt[r: r+patch_size, c: c+patch_size]
        Depth = depth[r: r+patch_size, c: c+patch_size]

        return  B,GT,Depth
