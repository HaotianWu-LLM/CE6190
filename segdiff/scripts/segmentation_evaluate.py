import sys
import random
sys.path.append(".")
sys.path.append('..')

import numpy
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import math
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion import  logger
from dataset.bratsloader2021 import BRATSDataset3D
import argparse

import collections
import logging
import math
import os
import time
from datetime import datetime
from sklearn.metrics import auc,roc_curve
from tqdm import tqdm

import dateutil.tz
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")


def precision(outputs,labels):
    TP = ((labels == 1) & (outputs == 1))
    FP = ((labels == 1) & (outputs == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)

def recall(outputs,labels):
    TP = ((labels == 1) & (outputs == 1))
    FN = ((labels == 0) & (outputs == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)

def FPR(outputs,labels):
    '假阳率'
    FP = ((labels == 1) & (outputs == 0))
    TN = ((labels == 0) & (outputs == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)

def ROC_AUC(outputs,labels):
    if type(labels) == torch.Tensor:
        return roc_curve(labels.detach().cpu().numpy().flatten(), outputs.detach().cpu().numpy().flatten())
    else:
        return roc_curve(labels.flatten(), outputs.flatten())
    
def AUC_score(fpr, tpr):
    return auc(fpr, tpr)


def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

# threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
def eval_seg(pred,true_mask_p,threshold = (0.5,)):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    对于每个阈值 th，函数将预测和真实标签都转换为二值图像（大于阈值的为1，否则为0）。
    对于每个阈值，计算每个分割目标的IoU和Dice系数，并累加结果。
    最后，将累加的结果除以阈值的数量，得到平均IoU和Dice系数
    '''
    b, c, h, w = pred.size()
    # 二通道问题
    if c == 2:
        # print('二通道...')
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            
            # 一个前景，一个后景
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    
    else:
        # print('单通道...')
        eiou, edice, prec_value, recall_value, auc_value = 0,0,0,0,0
        # 对每一个样本，按照多个threshold计算该样本最终的指标
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32') # (1,256,256)
            
            disc_mask = gt_vmask_p[:,0,:,:].squeeze(1).cpu().numpy().astype('int32') # (1,256,256)
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:].squeeze(1)).item()
            
            '''precision for torch'''
            prec_value += precision(vpred[:,0,:,:], gt_vmask_p[:,0,:,:].squeeze(1)).item()
            
            '''recall for numpy'''
            recall_value += recall(vpred[:,0,:,:], gt_vmask_p[:,0,:,:].squeeze(1)).item()
            
            '''auc for numpy'''
            fpr, tpr, _ = ROC_AUC(vpred[:,0,:,:], gt_vmask_p[:,0,:,:].squeeze(1))
            auc_value += AUC_score(fpr,tpr)
        # print('prec:{},recall:{},auc:{}'.format(prec_value,recall_value,auc_value))
            
        n = len(threshold)
        return eiou / n, edice / n, prec_value / n, recall_value / n, auc_value / n

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--test_data_dir")
    argParser.add_argument("--pred_data_dir")
    argParser.add_argument("--image_size",type=int,default=256)
    argParser.add_argument("--data_name")
    args = argParser.parse_args()
    
    logger.configure(dir = args.pred_data_dir)

    logger.log("creating data loader...")
    if args.data_name == 'ISIC':
        pass
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)
        test_ds = BRATSDataset3D(args.test_data_dir,'test',transforms=transform_train)
    else:
        print('不知道的数据')
        
    logger.log(f"====The length of test set is: {len(test_ds)}====")
    
    test_datal= torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False)

    mix_res = (0,0,0,0,0) # 用于储存IoU和Dice的计算结果
    num = 0 # 计数处理的文件个数
    # 读取数据
    for data,mask,path in tqdm(test_datal):
        gt = torch.unsqueeze(mask,0).float() / mask.max() # (1,1,1,256,256)
        num += 1
        
        # 找相应的预测图像
        # slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        slice_ID=path[0].split('.')[0]
      
        pred = Image.open(os.path.join(
     args.pred_data_dir,str(slice_ID)+'_output_ens'+".jpg"
        )) # (3,256,256)
        pred = torchvision.transforms.PILToTensor()(pred)
        pred = torch.unsqueeze(pred,0).float() 
        pred = pred / pred.max()
        
        # pred.shape=(1,3,256,256) gt.shape=(1,1,1,256,256)
        temp = eval_seg(pred,gt)
        mix_res = tuple([sum(a) for a in zip(mix_res,temp)])
        
        
    iou,dice,prec,recall,auc = tuple([a/num for a in mix_res])
    logger.log(f"iou is {iou}")
    logger.log(f"dice is {dice}")
    logger.log(f"precision is {prec}")
    logger.log(f"recall is {recall}")
    logger.log(f"auc is {auc}")
    
        


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
