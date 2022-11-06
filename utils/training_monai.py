import os
import sys
import math
import string
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn.init as init
from torchvision.utils import save_image
from torch.autograd import Variable
import monai
from scipy import ndimage
#from . import imgs as img_utils

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()
        
def save_weights(model, epoch, loss, WEIGHTS_PATH):
    weights_fname = 'weights-%d-%.3f.pth' % (epoch, loss)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH / 'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    ### removing module. from the keys
    for key in list(weights['state_dict'].keys()):
        if key.startswith("module."):
            weights['state_dict'][key[7:]] = weights['state_dict'].pop(key)
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {})"
          .format(startEpoch-1, weights['loss']))
    return model

def train_multi_LandD(writer, net, train_loader, train_ds, optimizer, scaler, criterion_2, epoch, lambda_2=1.0):

    epoch_loss = 0
    res = [0.29231285570766]*3

    # put the network in train mode; this tells the network and its modules to
    # enable training elements such as normalisation and dropout, where applicable
    net.train()
    for idx, batch_data in enumerate(train_loader):
        # move the data to the GPU
        inputs, label0, label1, label2 = batch_data['image'].cuda(), batch_data['label0'].cuda(), batch_data['label1'].cuda(), batch_data['label2'].cuda()
        labels_stacked = torch.stack((label0,label1,label2))
        labels = torch.squeeze(torch.permute(labels_stacked, (2, 1, 0, 3, 4, 5)))

        # prepare the gradients for this step's back propagation
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # run the network forwards
            outputs = net(inputs)

            loss2 = criterion_2(outputs, labels)
            loss =  lambda_2 * torch.sqrt(loss2)
            
        # compute the gradients
        if math.isnan(loss.item()):
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            optimizer.zero_grad()
            continue
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        epoch_loss += loss.item()
    
        
    monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=inputs, step=epoch, writer=writer, tag="input_image")
    monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=labels, step=epoch, writer=writer, tag="ground truth")
    monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=outputs, step=epoch, writer=writer, tag="output_image")
    
    epoch_loss /= (idx+1)
    return epoch_loss

def test_multi_LandD(net,val_loader, optimizer, criterion_2, lambda_2=1.0):

    # switch off training features of the network for this pass
    net.eval()
    test_loss = 0

    with torch.no_grad():
        # iterate over each batch of images and run them through the network in evaluation mode
        for idx, val_data in enumerate(val_loader):
            val_images, label0, label1, label2 = val_data['image'].cuda(), val_data['label0'].cuda(), val_data['label1'].cuda(), val_data['label2'].cuda()
            val_labels_stacked = torch.stack((label0,label1,label2))
            val_labels = torch.squeeze(torch.permute(val_labels_stacked, (2, 1, 0, 3, 4, 5)))

            # run the network
            val_pred = net(val_images)
            val_loss2 = criterion_2(val_pred, val_labels)
            val_loss =  lambda_2 * torch.sqrt(val_loss2)
            test_loss += val_loss.item()
        test_loss /= (idx+1)
    return test_loss



def train_LandD(writer, net, train_loader, train_ds, optimizer, scaler, criterion_2, epoch, lambda_2=1.0):

    epoch_loss = 0
    res = [0.29231285570766]*3

    # put the network in train mode; this tells the network and its modules to
    # enable training elements such as normalisation and dropout, where applicable
    net.train()
    for idx, batch_data in enumerate(train_loader):
        # move the data to the GPU
        inputs, labels = batch_data['image'].cuda(), batch_data['label'].cuda()

        # prepare the gradients for this step's back propagation
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # run the network forwards
            outputs = net(inputs)
            loss2 = criterion_2(outputs, labels)
            loss =  lambda_2 * torch.sqrt(loss2)

            
        # compute the gradients
        if math.isnan(loss.item()):
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            optimizer.zero_grad()
            print('loss is nan')
            continue
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        
        epoch_loss += loss.item()
        
    monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=inputs, step=epoch, writer=writer, tag="input_image")
    monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=outputs, step=epoch, writer=writer, tag="output_image")
    monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=labels, step=epoch, writer=writer, tag="ground truth")
    
    epoch_loss /= (idx+1)
    return epoch_loss

def test_LandD(net,val_loader, optimizer, criterion_2, lambda_2=1.0):

    # switch off training features of the network for this pass
    net.eval()
    test_loss = 0
    res = [0.29231285570766]*3

    # 'with torch.no_grad()' switches off gradient calculation for the scope of its context
    with torch.no_grad():
        # iterate over each batch of images and run them through the network in evaluation mode
        for idx, val_data in enumerate(val_loader):
            val_images, val_labels = val_data['image'].cuda(), val_data['label'].cuda()

            # run the network
            val_pred = net(val_images)
            val_loss2 = criterion_2(val_pred, val_labels)
            val_loss =  lambda_2 * torch.sqrt(val_loss2)
            test_loss += val_loss.item()

        test_loss /= (idx+1)
    return test_loss

def inferance_LandD(net,val_loader, optimizer, criterion_2, lambda_2=1.0):

    # switch off training features of the network for this pass
    net.eval()
    test_loss = 0
    res = [0.29231285570766]*3
    d1 = []
    d2 = []

    # 'with torch.no_grad()' switches off gradient calculation for the scope of its context
    with torch.no_grad():
        # iterate over each batch of images and run them through the network in evaluation mode
        for idx, val_data in enumerate(val_loader):
            val_images, val_labels = val_data['image'].cuda(), val_data['label'].cuda()

            ## run the network
            val_pred = net(val_images)
            val_loss2 = criterion_2(val_pred, val_labels)
            val_loss =  lambda_2 * torch.sqrt(val_loss2)
  
            ## Calculating center of mass after masking valeus less than maximum value 
            for i in range(val_pred.shape[0]):
                surr = np.ma.masked_less(val_pred.cpu().detach().numpy()[i,0,:,:,:],0.1*np.max(val_pred.cpu().detach().numpy()[i,0,:,:,:]))
                c1 = ndimage.measurements.center_of_mass(surr)
                surr = np.ma.masked_less(val_labels.cpu().detach().numpy()[i,0,:,:,:],0.1*np.max(val_labels.cpu().detach().numpy()[i,0,:,:,:]))
                c2 = ndimage.measurements.center_of_mass(surr)
                d1.append(np.sqrt(((c1[0]-c2[0])*res[0])**2 + ((c1[1]-c2[1])*res[1])**2 + ((c1[2]-c2[2])*res[2])**2))
                
            test_loss += val_loss.item()

        test_loss /= (idx+1)
    return test_loss, d1