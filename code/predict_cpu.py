#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
# from models.generator import Generator
from models.hat_arch import HAT
#from GanNet import SPANet
#Metrics lib
#from metrics import calc_psnr, calc_ssim
from numpy.random import RandomState
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default = 'demo',type=str)
#    parser.add_argument("--input_dir",default = r'../dataset/real_data/1/data/', type=str)
#    parser.add_argument("--input_dir",default = r'../../dataset/100H/test/data/', type=str)
#    parser.add_argument("--input_dir",default = r'../../dataset/rain800/test/data1/', type=str)
    parser.add_argument("--input_dir",default = r'../dataset1/100H/test/data/', type=str)
    parser.add_argument("--output_dir", default = r'./test_result/r1254/100H/epoch150/',type=str)
#    parser.add_argument("--gt_dir",default = r'../dataset/test/100H/gt/', type=str)
    args = parser.parse_args()
    return args

def align_to_four(img):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
#    print("img",img.shape)
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img




def predict(image,patch = False):
#    img_val = image
    
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image)
    # print("zheli1",image.shape)   #torch.Size([1, 3, 84, 128])
    
    bs, c, h, w = image.shape
    if patch == True:
#        patch_size = 480
        rand_state = RandomState(66)
        h_patch_size = int(h/1.50)
        w_patch_size = int(w/1.50)

        r = rand_state.randint(0, h - h_patch_size)
        c = rand_state.randint(0, w - w_patch_size)

        image = image[:, :, r: r+h_patch_size, c: c+w_patch_size]
#        gt = gt[r: r+h_patch_size, c: c+w_patch_size,:]
        print("imgshape",image.shape)

    with torch.no_grad():
#        A_, out_gan =  net_G(image)
        out_gan  =  net_G(image)
        # out_gan,_ , Attention1 =  net_G(image)
        print("zheli2")
        print("out_gan",out_gan.shape)
    # print("out_gan",out_gan.shape)


    
    out_gan = out_gan.cpu().data
    out_gan = out_gan.numpy()
    out_gan = out_gan.transpose((0, 2, 3, 1))
    out_gan = out_gan[0, :, :, :]*255.
    
    # Attention1 = Attention1.numpy()
    # Attention1 = Attention1.transpose((0, 2, 3, 1))
    # Attention1 = Attention1[0, :, :, :]*255.
    
    result = np.array(out_gan, dtype = 'uint8')
    # print("out_gan",out_gan.shape)
    # print(Attention1.shape)
#    cur_psnr = calc_psnr(result, gt)
#    cur_ssim = calc_ssim(result, gt)

#    return out_gan, cur_psnr, cur_ssim
    return out_gan#,Attention1


if __name__ == '__main__':
    args = get_args()


#net_G = Generator(feature_channels=128)
    
    G_path = r'./checkpoint/100H_1254/epoch150/hat_para.pth'
    net_G = HAT(overlap_ratio=0,upscale=1,upsampler='pixelshuffle',drop_path_rate=0.0)
#    net_G = torch.nn.DataParallel(net_G)
#    net_G.load_state_dict(torch.load(G_path))
#    net_G.eval()
    
    device = torch.device('cpu')
    state_dict = torch.load(G_path, map_location=device)
    from collections import OrderedDict
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        
#        name = k[7:]  # 去掉 `module.`
        name = k[7:]
        state_dict_new[name] = v
#        state_dict_new[name] = v
     
    net_G.load_state_dict(state_dict_new)
    net_G.eval()
    
    
    
    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
#        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        for i in range(num):

            img = cv2.imread(args.input_dir + input_list[i])
#            gt = cv2.imread(args.gt_dir + gt_list[i])
#            O_,_ , Attention1
            img = align_to_four(img)
#            gt = align_to_four(gt)
            print("imggg",img.shape)   #(84, 128, 3)
#            out_gan,cur_psnr, cur_ssim = predict(img,gt,patch = False)
            out_gan = predict(img,patch = False)
            # out_gan, Attention1 = predict(img,patch = False)
            

            
            
            img_name = input_list[i].split('.')[0]
#            sm = '%.4f' % cur_ssim
#            ps = '%.4f' % cur_psnr
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
#            cv2.imwrite(args.output_dir + img_name +'sm' + str(sm) +'ps' + str(ps) + 'gan.jpg', out_gan)
            cv2.imwrite(args.output_dir + img_name + '_derain.jpg', out_gan)
            # cv2.imwrite(args.output_dir + img_name + 'att.jpg', Attention1)
            print(",",args.output_dir)
    else:
        print ('Mode Invalid!')
