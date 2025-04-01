import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import cv2
from torch.optim.lr_scheduler import MultiStepLR
from loss import *
#from cal_ssim import SSIM
from SSIM import SSIM
import time
import numpy as np
import cv2
import random
import time
import os
import argparse

from models.CAMixerSR_attfrth import CAMixerSR

from func import *

from data.dataset_util_stage2 import RainDataset
from torch.utils.data import DataLoader



class trainer:
    def __init__(self, opt):

        self.net_G = CAMixerSR(scale=1, n_feats=60, ratio=0.5)

        self.net_G  = torch.nn.DataParallel(self.net_G)
        print('# generator parameters:', sum(param.numel() for param in self.net_G.parameters()))
#        self.optim1 = torch.optim.Adam(filter(lambda p : p.requires_grad, self.net_G.parameters()), lr = opt.lr, betas = (0.5,0.99))   #之前的，好像说可以初始化之类的
        self.optim1 = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr)
        self.sche = MultiStepLR(self.optim1, milestones=[30, 50, 80, 120], gamma=0.2) #调整学习率
        self.iter = opt.iter
        self.batch_size = opt.batch_size
        self.out_path = opt.result
        print('Loading dataset ...\n')
#        train_dataset = Dataset(data_path=opt.train_dataset)    #成对返回雨、无雨  DerainDataset.py中
#        valid_dataset = Dataset(data_path=opt.eval_dataset)
        
        train_dataset = RainDataset(opt)   #list成对返回雨、无雨
        valid_dataset = RainDataset(opt, is_eval=True)
        train_size = len(train_dataset)
        valid_size = len(valid_dataset)
        
        
        # print("# train set : {}".format(train_size))
        # print("# eval set : {}".format(valid_size))

        
        self.train_loader = DataLoader(train_dataset, num_workers=0, batch_size=opt.batch_size,shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, num_workers=0,batch_size=opt.batch_size)

        print("# train set : {}".format(train_size))
        print("# eval set : {}".format(valid_size))
              

        #输出计算ssim
        #self.ssim = SSIM()
        self.ssim = SSIM().cuda()
#        #Attention Loss
        # self.criterionAtt = AttentionLoss(theta=0.8, iteration=2)
        #Perceptual Loss
        self.criterionPL = PerceptualLoss()
        #Multiscale Loss
#        self.criterionML = MultiscaleLoss(ld = [0.6,0.8,1.0],batch=self.batch_size)
        #MAP Loss
#        self.criterionMAP = MAPLoss(gamma = 0.05)
        	#MSE Loss
#        self.criterionMSE = nn.MSELoss().cuda()
        #self.criterionMSE = nn.MSELoss()
        #self.criterionATT = nn.MSELoss()
        self.criterionMSE = nn.MSELoss().cuda()
        self.criterionATT = nn.MSELoss().cuda()
        self.criterionEDGE = EDGELoss()
        
    def forward_process(self,I, GT, is_train=True):

        M_ = []
        for i in range(I.shape[0]):
            M_.append(get_mask(np.array(I[i]),np.array(GT[i])))  # #bs中每个图求mask，get_mask在func.py中
        M_ = np.array(M_)
        M_ = torch_variable(M_, is_train)
        
        I_ = torch_variable(I, is_train)
        GT_ = torch_variable(GT, is_train)

        
        if is_train:
            self.net_G.train()
            # O_,_ , Attention1 = self.net_G(I_,)  #分别为derain, derain_list
            # O_= self.net_G(I_,)[0]
            Attention1, O_, _, = self.net_G(I_)

            # print("O_O_", len(O_))
#            loss
            # loss_MSE =self.criterionMSE(O_ ,GT_.detach())
            # print("O_O_1", O_.shape, GT_.detach_().shape)
            loss_MSE =self.criterionMSE(O_ ,GT_.detach_())
            # assert 1==0

            loss_ATT = self.criterionATT(Attention1,M_.detach_())
#            perceptual_loss 
            loss_PL = self.criterionPL(O_, GT_.detach_())
#            SSIM_loss
            ssim_loss = 1-self.ssim(O_,GT_.detach_())
            # assert 1==0
            edge_loss = self.criterionEDGE(O_,GT_.detach_())
            loss_G = loss_MSE + ssim_loss + 0.1*edge_loss + loss_PL+0.1*loss_ATT
#            loss_G = loss_MSE + 0.5*loss_PL   + ssim_loss
            output = [loss_G, O_, loss_MSE, ssim_loss, edge_loss, loss_PL, M_]  
            # assert 1==0

        else: # validation
            self.net_G.eval()
            # O_,_  = self.net_G(I_)
            Attention1,O_,_  = self.net_G(I_)
            output = O_
            self.net_G.train()
        return output

    def train_start(self):
#        loss_sum = 0.
#        valid_loss_sum = 0.
#        f=open('train_process.txt','w')

        count = 0
        for epoch in range(0, self.iter+1):   #self.iter=200
            f=open('train_process.txt','a')
            since = time.time()
            self.sche.step()
            lr = self.optim1.param_groups[0]['lr']
            print('%d_epochGGlearning rate = %.7f' % (epoch,lr))

            for i, data in enumerate(self.train_loader):
#                print("[epoch %d][%d/%d] " %(epoch, i+1, len(self.train_loader)))
                count+=1
                I_, GT_ = data     #
#                print("I_, GT_,",I_.shape, GT_.shape,)
#                print("I_, GT_,",I_.requires_grad, GT_.requires_grad,)
                loss_G, O_, loss_MSE, ssim_loss, edge_loss, loss_PL, mask_target = self.forward_process(I_,GT_)
                
#                print("lossg",loss_G.item())
#                print("loss_MSE,ssim_loss",loss_MSE.item(),ssim_loss.item())
#                print("edge_loss,loss_ATT",edge_loss.item(),loss_ATT.item())
#                print("loss_att,edge_loss",loss_att.item(),edge_loss.item())
                

                self.optim1.zero_grad()
                loss_G.backward()
                self.optim1.step()  #参数更新
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                           time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
            strattime = str(epoch)
            f.write(strattime)
            f.write('      ')
            f.write('loss_MSE')
            f.write('      ')
            mesloss = str(loss_MSE.item())
            f.write(mesloss)
            f.write('      ')
            f.write('ssim_loss')
            f.write('      ')
            ssimloss = str(ssim_loss.item())
            f.write(ssimloss)
            f.write('      ')
    
            f.write('edge_loss')
            f.write('      ')
            edgeloss = str(edge_loss.item())
            f.write(edgeloss)
            f.write('      ')
            
            # f.write('loss_ATT')
            # f.write('      ')
            # lossATT = str(loss_ATT.item())
            # f.write(lossATT)
            # f.write('      ')
            
            f.write('loss_PL')
            f.write('      ')
            lossPL = str(loss_PL.item())
            f.write(lossPL)
            f.write('\r\n')
            f.close()
            if epoch % 5 ==0:
                
                where_to_save = self.out_path
                where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch) + '/')    #拼接
                if not os.path.exists(where_to_save_epoch):     #判断文件是否存在
                    os.makedirs(where_to_save_epoch)   #用于递归创建目录
                file_name = os.path.join(where_to_save_epoch, 'hat_para.pth')
                torch.save(self.net_G.state_dict(), file_name,  _use_new_zipfile_serialization=False) #仅保存和加载参数
                mask_target = mask_target.cpu().data.numpy().transpose((0, 2, 3, 1))
                cv2.imwrite(where_to_save_epoch + 'att.png', mask_target[0]*255)



        return
    
    
    
#net_G=Generator(Feature_dim=32, recurrent_iter=4)
#print('# generator parameters:', sum(param.numel() for param in net_G.parameters()))