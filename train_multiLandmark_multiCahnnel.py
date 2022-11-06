import time
import os
import glob
import numpy as np
import datetime
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import utils.training_monai as train_utils


from src import SaveStateReader, data_import, data_preprocessing

from monai.utils import first
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    Flipd,
    RandAffined,
)

from monai.networks.nets import HighResNet
from monai.data import DataLoader, Dataset
import nibabel as nib

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import timedelta

### NN output & logging ###
if not os.path.exists('.log'):
    os.makedirs('.log')
    os.makedirs('.results')
if not os.path.exists('.weights'):
    os.makedirs('.weights')
LOG_PATH = Path('./.log/')


current_date = datetime.datetime.now()-timedelta(hours=4)
train_file = "train_highresnet_run_{}_{}_{}_{}".format(
    current_date.month, current_date.day, current_date.hour, current_date.minute)

log_root = Path(LOG_PATH) / train_file
print("log_root:  ", log_root, '\n')



### Path setup ###
## training data path
train_data_path = '/home/ec2-user/SageMaker/autodetect/data_augmented/train_data'
train_data_raw = '/home/ec2-user/SageMaker/autodetect/data/train/aneurysm_dicom'
## Validation data path
val_data_path = '/home/ec2-user/SageMaker/autodetect/data_augmented/val_data'
val_data_raw = '/home/ec2-user/SageMaker/autodetect/data/val/aneurysm_dicom'
## Test data path

test_data_path = "/home/ec2-user/SageMaker/autodetect/data_augmented/test_data"
test_data_raw = '/home/ec2-user/SageMaker/autodetect/data/test/aneurysm_dicom'

affix = '_augmented_multiLM' ## multi landmark in different output channel, if there are multiple landmark in one output channel then use train.py but need to change the preprocessing step to generate the data from src/data_preprocessing

batch_size = 4
learning_rate = 1e-4*0.1
weight_decay = 1e-5
N_EPOCHS = 250
torch.cuda.manual_seed(0)
num_landmark = 3

use_pretrained = False
model_choice = "unet2"
## There is not pretrained weight for this case

# model_choice = "unet2" ### unet1, unet2, and unet3 are small, medium, and large unet models, respectively.
# assert model_choice in ["unet1", "unet2", "unet3", "highresnet"], "Choose a model between unet1, unet2, unet3, and highresnet "
# fpath = Path('/home/ec2-user/SageMaker/autodetect/pretrained_weights') / f'latest_{model_choice}.th'

## initial augmentation
for path in zip([train_data_path,val_data_path],[train_data_raw,val_data_raw]):
    if not os.path.exists(path[0]+affix):
        crop_size = [60,80,100,200]
        ax_orders = ['xzy','zyx','zxy','yzx']
        ## Creating an augmentation object
        data_augmentation = data_preprocessing.preprocessing_aug(path[1],image_size=(64, 64, 64),train=True)
        ## generating the augmented images with different cropsize and axes order 
        counter = data_augmentation.generate_image_label_multi_landmark(num_landmark,path[0]+affix,crop_size=crop_size,ax_orders=ax_orders,sigma=3)
        print(f"{counter} images was found and {len(crop_size)*(len(ax_orders)+1)*counter} images were generated and saved in {path[0]+affix}.")
        
## Generating the heatmap and input data for test 
for idx_path, path in enumerate(zip([test_data_path],[test_data_raw])):
    if not os.path.exists(path[0]+affix):
        crop_size = [60,80,100,200]
        ax_orders = ['xzy','zyx','zxy','yzx']
        ## Creating an augmentation object
        data_augmentation = data_preprocessing.preprocessing_aug(path[1],image_size=(64, 64, 64),train=False)
        ## generating the augmented images with different cropsize and axes order 
        counter = data_augmentation.generate_image_label_multi_landmark(num_landmark,path[0]+affix,sigma=3)
        print(f"{counter} images were generated for test and saved in {path[0]+affix}.")


## Augmentation
generat_transforms = Compose(
    [
        LoadImaged(keys=["image", "label0", "label1", "label2"]),
        AddChanneld(keys=["image", "label0", "label1", "label2"]),
        RandAffined(keys=["image", "label0", "label1", "label2"], prob=0.5, translate_range=10), 
        RandRotated(keys=["image", "label0", "label1", "label2"], prob=0.5, range_x=1.0, range_y=1.0, range_z=1.0),
        #RandGaussianNoised(keys='image', prob=0.5),
        ToTensord(keys=["image", "label0", "label1", "label2"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label0", "label1", "label2"]),
        AddChanneld(keys=["image", "label0", "label1", "label2"]),
        ToTensord(keys=["image", "label0", "label1", "label2"]),
    ]
)

train_labels = []
train_labels.append(glob.glob(train_data_path+affix+f'/*0blob.npy'))
for LM in range(num_landmark-1):
    train_lbl = []
    for p in train_labels[0]:
        p = p[:-9]
        p = p+f'{LM+1}blob.npy'
        train_lbl.append(p)
    train_labels.append(train_lbl)
    
train_images = []   
for p in train_labels[0]:
    p = p[:-9]
    p = p+'.npy'
    train_images.append(p)
train_files = [{"image": image, 'label0': l0, 'label1': l1,'label2': l2} for image,l0,l1,l2 in zip(train_images,train_labels[0],train_labels[1],train_labels[2])]
train_ds = Dataset(data=train_files, transform=generat_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)


val_labels = []
val_labels.append(glob.glob(val_data_path+affix+'/*0blob.npy'))
for LM in range(num_landmark-1):
    val_lbl = []
    for p in val_labels[0]:
        p = p[:-9]
        p = p+f'{LM+1}blob.npy'
        val_lbl.append(p)
    val_labels.append(val_lbl)

val_images = []
for p in val_labels[0]:
    p = p[:-9]
    p = p+'.npy'
    val_images.append(p)
    
val_files = [{"image": image, 'label0': l0, 'label1': l1,'label2': l2} for image,l0,l1,l2 in zip(val_images, val_labels[0], val_labels[1], val_labels[2])]
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=10)


writer = SummaryWriter()

RESULTS_PATH = Path('.results/') / train_file
WEIGHTS_PATH = Path('.weights/') / train_file
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)

### highresnet setup
DEFAULT_LAYER_PARAMS_3D = (
    # initial conv layer
    {"name": "conv_0", "n_features": 16, "kernel_size": 3},
    # residual blocks
    {"name": "res_1", "n_features": 16, "kernels": (3, 3), "repeat": 3},
    {"name": "res_2", "n_features": 32, "kernels": (3, 3), "repeat": 3},
    {"name": "res_3", "n_features": 64, "kernels": (3, 3), "repeat": 3},
    # final conv layers
    {"name": "conv_1", "n_features": 80, "kernel_size": 1},
    {"name": "conv_2", "kernel_size": 1},)


### Model

if model_choice == "unet3":
    # Unet3 (large size)
    model = tiramisu.FCDenseNet_gray_seg_landmark3(n_classes=1).cuda()
elif model_choice == "unet2":
    # Unet2 (medium size)
    model = tiramisu.FCDenseNet_gray_seg_landmark2(n_classes=1).cuda()
elif model_choice == "highresnet":
    # highresnet model
    model = HighResNet(spatial_dims=3, in_channels=1, out_channels=1,dropout_prob=0.2,layer_params=DEFAULT_LAYER_PARAMS_3D)
else:
    # Unet1 (small size)
    model = tiramisu.FCDenseNet_gray_landmark1(n_classes=1).cuda()oss_CE = nn.CrossEntropyLoss().cuda()

model.apply(train_utils.weights_init)
if use_pretrained:
    model = train_utils.load_weights(model, fpath)
loss_MSE = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(model.parameters(), learning_rate)
optimizer=  torch.optim.RMSprop(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10, factor=0.9,min_lr=1e-7*1.0)

### Multi-GPU ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model) 
model.to(device)
print(sum(p.numel() for p in model.parameters()))


for epoch in range(1, N_EPOCHS+1):
    print('epoch: ', epoch)

    ### Train ###
    since = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    ## multiple output channel
    trn_loss = train_utils.train_multi_LandD(writer, model, train_loader,train_ds, optimizer, scaler, loss_MSE, epoch, lambda_2=50.0)

    print('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss))
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.cuda.empty_cache()
    
    ### Validation ###
    since = time.time()
    
    ## multiple output channel
    val_loss = train_utils.test_multi_LandD(model,val_loader, optimizer, loss_MSE, lambda_2=50.0)
    scheduler.step(val_loss)
    
    print(f"lr: {optimizer.param_groups[0]['lr']}")
    print('Val Synthetic - Loss: {:.4f}'.format(val_loss))
    time_elapsed = time.time() - since  
    print('Validation Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    torch.cuda.empty_cache()
    
    ## There is no inference for this case yet
    
    train_utils.save_weights(model, epoch, val_loss, WEIGHTS_PATH)
writer.close()


    