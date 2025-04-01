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
        # print("self.dataset 1",self.dataset)
        # assert 1==0
        # dataset = open(self.dataset, 'r').read().split()
        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
#        self.depth_list = sorted(glob.glob(self.dataset+'/depth_gen/*'))
#        self.att_list = sorted(glob.glob(self.dataset+'/att_gen/*'))
        self.rand_state = RandomState(66)   #是不是相当于seed
        self.patch_size = opt.patch_size
   
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]
#        depth_name = self.depth_list[idx]
#        att_name = self.att_list[idx]
#        print("img_name",self.img_list)
#        print("gt_name",self.gt_list)
#        print("att_name",self.att_list)
#        img_name = img_name[i].split('.')[0]
        # img_name_t = img_name.split('-')[-1]
        # gt_name_t = gt_name.split('-')[-1]
#        print('img_name',img_name)
#        print('gt_name',gt_name)
        img_name_t = img_name.split('/')[-1]
        gt_name_t = gt_name.split('/')[-1]
#        print('img_name_t',img_name_t)
#        print('gt_name_t',gt_name_t)
#        print("img_name_t",img_name)
#        print("gt_name_t",gt_name)
#        print("depth_name_t",depth_name)
#        print("att_name_t",att_name)
#        print("d",att_name_t == depth_name_t)
#        assert img_name_t == gt_name_t,'gt_name不匹配'
#        assert img_name_t == depth_name_t,'depth_name不匹配'
#        assert img_name_t == att_name_t,'att_name不匹配'
        if img_name_t != gt_name_t:
            print('img_name',img_name)
            print('gt_name',gt_name)
            raise AssertionError('gt_name不匹配')
#        if img_name_t != depth_name_t:
#            print('img_name',img_name)
#            print('depth_name',depth_name)
#            raise AssertionError('depth_name不匹配')
#        if img_name_t != att_name_t:
#            print('img_name',img_name)
#            print('att_name',att_name)
#            raise AssertionError('att_name不匹配')

        img = cv2.imread(img_name,-1)
        gt = cv2.imread(gt_name,-1)   #以原保存方式读入  （灰度还是彩色）
#        depth = cv2.imread(depth_name,-1)
#        print("读入depth",depth.shape)
#        att = cv2.imread(att_name,-1)
#        att = cv2.imread(att_name,-1)
#        print("读入att",att.shape)
        img , gt  = self.crop(img, gt)  #crop
        if img.dtype == np.uint8:   #调用这里
            img = (img / 255.0).astype('float32')
#            print("归一化")
        if gt.dtype == np.uint8:
            gt = (gt / 255.0).astype('float32')
#        if depth.dtype == np.uint8:
#            depth = (depth / 255.0).astype('float32')
###            print("从图像转为floast",depth.dtype)
#            distance = -np.log(depth + 0.01)
#            distance = distance/np.max(distance)  #归一化
#            _range = np.max(distance) - np.min(distance)
#            if _range == 0:
##            if _range > 0:
#                if np.max(distance) > float(0):
#                    print ("都在远处,distance[0,0]",distance[0,0],np.max(distance))
#                    distance = np.ones(distance.shape)
#                    distance = (distance*0.9).astype('float32')
##                    distance = distance.astype('float32')
##                    print("从图像转为floast",distance.dtype)
#                elif np.max(distance) < float(0):
#                    print ("都在近处,distance[0,0]",distance[0,0],np.max(distance))
#                    distance = np.ones(distance.shape)
#                    distance = (distance*0.01).astype('float32')
#                else:
#                    print("太巧了,depth都在0.99,distance[0,0]",distance[0,0])
#                    distance = np.ones(distance.shape)
#                    distance = (distance*0.01).astype('float32')
#            else:
##                print("if _range == 0:")
#                distance = (distance - np.min(distance)) / _range

#        if att.dtype == np.uint8:   #调用这里
#            att = (att / 255.0).astype('float32')
            
#        print("读入维度",img.shape, gt.shape, distance.shape, att.shape)

        return [img,gt]

    def crop(self, img_pair,gt):
        patch_size = self.patch_size
        h, w, c = img_pair.shape
        r = self.rand_state.randint(0, h - patch_size)
        c = self.rand_state.randint(0, w - patch_size)
        B = img_pair[r: r+patch_size, c: c+patch_size]
        GT = gt[r: r+patch_size, c: c+patch_size]
#        Depth = depth[r: r+patch_size, c: c+patch_size]
#        Att = att[r: r+patch_size, c: c+patch_size]

        return  B,GT
