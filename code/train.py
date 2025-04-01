import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import cv2
from options.train_options import TrainOptions
from train_module import trainer
from data.dataset_util import RainDataset
from torch.utils.data import DataLoader
from models import*
from models.dense import ConvGRU
#print("ConvGRU",ConvGRU)
opt = TrainOptions().parse() #继承baseoption并执行parse()方法 

tr = trainer(opt)
tr.train_start()

 