# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:02:21 2023

@author: shijigang
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:21:01 2023

@author: shijigang
"""
import math
import numpy as np
from data_generate import data_generate
# import tensorflow as tf 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from torch.utils import data
import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn as nn
from losses import SupConLoss
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    train_on_gpu=True    #控制是否使用gpu
    print(" gpu is used for training")
    
def data_loader(database='PTB_XL',model_type='VGG_Net',label_rate=0.1):

    batchsize_all=256
    data_labeled,label_labeled,data_unlabeled_u1,data_unlabeled_u2,data_unlabeled_strong,x_test,y_test,num_classes=data_generate(database=database,label_rate=label_rate)
    
    batchsize_x=int(batchsize_all//(1/label_rate))
    batchsize_u=int(batchsize_all-batchsize_x)

    batchsize_test=256
    
    class Labeled_TrainDataset(data.Dataset):
        def __init__(self):
            self.Data = data_labeled
            self.Label = label_labeled
    
        def __getitem__(self, index):
            txt = torch.from_numpy(self.Data[index])  # from_numpy 只能是numpy array！！！
            label = torch.tensor(self.Label[index])
            return txt, label
    
        def __len__(self):
            return len(self.Data)
        
    class Unlabeled_TrainDataset(data.Dataset):
        def __init__(self):
            self.Data_u1 = data_unlabeled_u1
            self.Data_u2 = data_unlabeled_u2
            self.Data_strong = data_unlabeled_strong

        def __getitem__(self, index):
            txt_u1 = torch.from_numpy(self.Data_u1[index])  # from_numpy 只能是numpy array！！！
            txt_u2 = torch.from_numpy(self.Data_u2[index])  # from_numpy 只能是numpy array！！！
            txt_strong = torch.from_numpy(self.Data_strong[index])  # from_numpy 只能是numpy array！！！
            return txt_u1,txt_u2,txt_strong
    
        def __len__(self):
            return len(self.Data_u1)
    
    
    class TestDataset(data.Dataset):
        def __init__(self):
            self.Data = x_test
            self.Label = y_test
    
        def __getitem__(self, index):
            txt = torch.from_numpy(self.Data[index])  # from_numpy 只能是numpy array！！！
            label = torch.tensor(self.Label[index])
            return txt, label
    
        def __len__(self):
            return len(self.Data)
        
    test = TestDataset()
    test_loader = data.DataLoader(test, batch_size=batchsize_test,shuffle=True)  #将数据集进行划分
    labeled_train = Labeled_TrainDataset()
    labeled_trainloader = data.DataLoader(labeled_train, batch_size=batchsize_x,shuffle=True)
    unlabeled_train = Unlabeled_TrainDataset()
    unlabeled_trainloader = data.DataLoader(unlabeled_train, batch_size=batchsize_u,shuffle=True)
    
    
    
    return labeled_trainloader,unlabeled_trainloader,test_loader,num_classes
    