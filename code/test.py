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
#from models.dense import ConvGRU
#print("ConvGRU",ConvGRU)

#from models.my_modify_densenet import DenseNet3
from models.generator_rnn import Generator
#from models.generator_rnn import Generator
from func import *
#from data.dataset_util import RainDataset   #bu yong zhiqian de le
#from data.DerainDataset import *
from data.dataset_util_stage2 import RainDataset
from torch.utils.data import DataLoader

#from tensorboardX import SummaryWriter

class trainer:
    def __init__(self, opt):
        self.net_G = Generator(Feature_dim=32, depth=50, growth_rate=16, bottleneck=True,dropRate = 0, recurrent_iter=4).cuda()
        self.net_G  = torch.nn.DataParallel(self.net_G)
        print('# generator parameters:', sum(param.numel() for param in self.net_G.parameters()))
#        self.optim1 = torch.optim.Adam(filter(lambda p : p.requires_grad, self.net_G.parameters()), lr = opt.lr, betas = (0.5,0.99))   #之前的，好像说可以初始化之类的
        self.optim1 = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr)
        self.sche = MultiStepLR(self.optim1, milestones=[50, 100, 150], gamma=0.1)
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
        self.train_loader = DataLoader(train_dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, num_workers=0,batch_size=opt.batch_size)

        print("# train set : {}".format(train_size))
        print("# eval set : {}".format(valid_size))
              

        #输出计算ssim
        self.ssim = SSIM().cuda()
#        #Attention Loss
#        self.criterionAtt = AttentionLoss(theta=0.8, iteration=2)
        #Perceptual Loss
        self.criterionPL = PerceptualLoss()
        #Multiscale Loss
        self.criterionML = MultiscaleLoss(ld = [0.6,0.8,1.0],batch=self.batch_size)
        #MAP Loss
#        self.criterionMAP = MAPLoss(gamma = 0.05)
        	#MSE Loss
        self.criterionMSE = nn.MSELoss().cuda()
        
    def forward_process(self,I, GT, is_train=True):
#        I_ = I_.cuda()
#        GT_ = GT_.cuda()
        I_ = torch_variable(I, is_train)
        GT_ = torch_variable(GT, is_train)
#        Depth = Depth.unsqueeze(1)
#        Depth_ = Variable(Depth).cuda()
        
        if is_train:
            self.net_G.train()
            O_,_  = self.net_G(I_,)  #分别为3个网络的去雨图

            loss_MSE =self.criterionMSE(O_ ,GT_)
            
            #perceptual_loss O_: generation, T_: GT  去除per loss
#            loss_PL = self.criterionPL(O_, GT_)

#            SSIM_loss
            ssim_loss = 1-self.ssim(O_,GT_)
            
            loss_G = loss_MSE + ssim_loss
#            loss_G = loss_MSE + 0.5*loss_PL   + ssim_loss
            output = [loss_G, O_]  #loss_MAP是对D的att loss 判断

        else: # validation
            self.net_G.eval()
            O_,_  = self.net_G(I_)
            output = O_
            self.net_G.train()
        return output

    def train_start(self):
        loss_sum = 0.
        valid_loss_sum = 0.

        count = 0
        for epoch in range(0, self.iter+1):   #self.iter=200
            since = time.time()
            self.sche.step()
            lr = self.optim1.param_groups[0]['lr']
            print('%d_epochGGlearning rate = %.7f' % (epoch,lr))

            for i, data in enumerate(self.train_loader):
                print("[epoch %d][%d/%d] " %(epoch, i+1, len(self.train_loader)))
                count+=1
                I_, GT_ = data     #Depth_已经转化为距离，远的大，近的小
#                print("I_, GT_,",I_.shape, GT_.shape,)
#                print("I_, GT_,",I_.requires_grad, GT_.requires_grad,)
                loss_G, O_ = self.forward_process(I_,GT_)

                self.optim1.zero_grad()
                loss_G.backward()
                self.optim1.step()

#                测试每次迭代每组输入是否成对
#                where_to_save = self.out_path
#                where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch) + '/'+ "iter" +str(i) + '/') 
#                if not os.path.exists(where_to_save_epoch):
#                    os.makedirs(where_to_save_epoch)
#                print("where_to_save_epoch",where_to_save_epoch)
                
#                img = I_[0].cpu().data
#                img = img.detach().numpy()
##                img = img.copy()
#                img = img.transpose((1, 2, 0))
#                img = 255*img
#                img_new = np.zeros(img.shape)
#                img_new[:,:,0]= img[:,:,2]
#                img_new[:,:,1]= img[:,:,1]
#                img_new[:,:,2]= img[:,:,0]
#                cv2.imwrite(where_to_save_epoch +str(i) + 'rain.jpg',img_new)
#                
#                img1 = GT_[0].cpu().data
#                img1 = img1.detach().numpy()
##                img1 = img1.copy()
#                img1 = img1.transpose((1, 2, 0))
#                img1 = 255*img1
#                img1_new = np.zeros(img1.shape)
#                img1_new[:,:,0]= img1[:,:,2]
#                img1_new[:,:,1]= img1[:,:,1]
#                img1_new[:,:,2]= img1[:,:,0]
#                cv2.imwrite(where_to_save_epoch +str(i) + 'GT.jpg',img1_new)
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                           time_elapsed // 60, time_elapsed % 60)) # 打印出来时间

#            for i, data in enumerate(self.valid_loader):
#                I_, GT_img  = data    #已经resize过了
#                I_ = I_.cuda()
#                GT_img = GT_img.cuda()
#
#
#                output= self.forward_process(I_,GT_img, is_train=False)#
#
#                where_to_save = self.out_path
#                where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch)  + '/') 
##                print("save path",where_to_save_epoch)
#                if not os.path.exists(where_to_save_epoch):
#                    os.makedirs(where_to_save_epoch)
#
##                GT_img = torch_variable(GT_, is_train = False)
##                print("output",output.shape)  #([2, 3, 100, 100])
##                print("GT_img",GT_img.shape)  #([2, 3, 100, 100])
#                
#                for j in range (0,output.shape[0]):
#                    
#                    sm = self.ssim(output[j].unsqueeze(0),GT_img[j].unsqueeze(0))
##                    print("ssim_l",output[j].unsqueeze(0))
#                    sm = '%.4f' % sm
##                    print("ssim_lll",sm)
#                    img = I_[j].cpu().data
#                    img = img.detach().numpy()
##                    img = img.copy()
#                    img = img.transpose((1, 2, 0))
#                    
#                    img = 255*img
#                    img_new = np.zeros(img.shape)
#                    img_new[:,:,0]= img[:,:,2]
#                    img_new[:,:,1]= img[:,:,1]
#                    img_new[:,:,2]= img[:,:,0]
#                    cv2.imwrite(where_to_save_epoch +str(i) + "it"+ str(j)  +'rain.jpg',img_new)
#                    
#                    img2 = output[j].cpu().data
#                    img2 = img2.detach().numpy()
##                    img2 = img2.copy()
#                    img2 = img2.transpose((1, 2, 0))
#                    img2 = 255*img2
#                    
#                    img2_new = np.zeros(img2.shape)
#                    img2_new[:,:,0]= img2[:,:,2]
#                    img2_new[:,:,1]= img2[:,:,1]
#                    img2_new[:,:,2]= img2[:,:,0]
##                    cv2.imwrite(where_to_save_epoch +str(i) + "it"+ str(j) +'de2.jpg',img2)
#                    cv2.imwrite(where_to_save_epoch +str(i) + "it"+ str(j)+"sm" +str(sm)+'de2.jpg',img2_new)
#
#                    img3 = GT_img[j].cpu().data  
#                    img3 = img3.detach().numpy()
##                    img3 = img3.copy()
#                    img3 = img3.transpose((1, 2, 0))
#                    img3 = 255*img3
#                    img3_new = np.zeros(img3.shape)
#                    img3_new[:,:,0]= img3[:,:,2]
#                    img3_new[:,:,1]= img3[:,:,1]
#                    img3_new[:,:,2]= img3[:,:,0]
#                    cv2.imwrite(where_to_save_epoch +str(i) + "it"+ str(j)+'gt.jpg',img3_new)

            where_to_save = self.out_path
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch) + '/') 
            if not os.path.exists(where_to_save_epoch):
                os.makedirs(where_to_save_epoch)
            file_name = os.path.join(where_to_save_epoch, 'gene_para.pth')
            torch.save(self.net_G.state_dict(), file_name)



        return
    
    
    
net_G=Generator(Feature_dim=32, recurrent_iter=4)
print('# generator parameters:', sum(param.numel() for param in net_G.parameters()))
input = torch.rand(1,3,64,64)
##mask = torch.rand(1,1,224,224).cuda()
output = net_G(input)