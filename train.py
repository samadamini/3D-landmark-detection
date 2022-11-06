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
from models import tiramisu


from src import SaveStateReader,data_import, data_preprocessing

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

writer = SummaryWriter()

RESULTS_PATH = Path('.results/') / train_file
WEIGHTS_PATH = Path('.weights/') / train_file
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)


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

affix = '_augmented'

batch_size = 4
learning_rate = 1e-4*0.5
weight_decay = 1e-5
N_EPOCHS = 200
torch.cuda.manual_seed(0)

## pretraining setup
use_pretrained = True
model_choice = "unet2" ### unet1, unet2, and unet3 are small, medium, and large unet models, respectively.
assert model_choice in ["unet1", "unet2", "unet3", "highresnet"], "Choose a model between unet1, unet2, unet3, and highresnet "
fpath = Path('/home/ec2-user/SageMaker/autodetect/pretrained_weights') / f'latest_{model_choice}.th'


## initial augmentation for training and validation + Generating the heatmap and input data
for idx_path, path in enumerate(zip([train_data_path,val_data_path],[train_data_raw,val_data_raw])):
    if not os.path.exists(path[0]+affix):
        crop_size = [60,80,100,200]
        ax_orders = ['xzy','zyx','zxy','yzx']
        ## Creating an augmentation object
        data_augmentation = data_preprocessing.preprocessing_aug(path[1],image_size=(64, 64, 64),train=True)
        ## generating the augmented images with different cropsize and axes order 
        counter = data_augmentation.generate_image_label(path[0]+affix,crop_size=crop_size,ax_orders=ax_orders,sigma=4)
        print(f"{counter} images was found and {len(crop_size)*(len(ax_orders)+1)*counter} images were generated for train and val and saved in {path[0]+affix}.")

## Generating the heatmap and input data for test 
for idx_path, path in enumerate(zip([test_data_path],[test_data_raw])):
    if not os.path.exists(path[0]+affix):
        ## Creating an augmentation object
        data_augmentation = data_preprocessing.preprocessing_aug(path[1],image_size=(64, 64, 64),train=False)
        ## generating the augmented images with different cropsize and axes order 
        counter = data_augmentation.generate_image_label(path[0]+affix,sigma=4)
        print(f" {counter} images were generated for test and saved in {path[0]}.")


## Augmentation
generat_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=1.0, range_y=1.0, range_z=1.0),
        #RandGaussianNoised(keys='image', prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ]
)

## data loader for training
train_labels = glob.glob(train_data_path+affix+'/*blob.npy')
train_images = []
for p in train_labels:
    p = p[:-8]
    p = p+'.npy'
    train_images.append(p)
train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
train_ds = Dataset(data=train_files, transform=generat_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)

## data loader for validation
val_labels = glob.glob(val_data_path+affix+'/*blob.npy')
val_images = []
for p in val_labels:
    p = p[:-8]
    p = p+'.npy'
    val_images.append(p)
val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=10)

## data loader for test
test_labels1 = glob.glob(test_data_path+affix+'/*blob.npy') ## finding all data in test dataset
test_labels2 = glob.glob(val_data_path+affix+'/*200xyzblob.npy')  ## finding original volumes without cropping oe swapped axes from validation
test_labels = test_labels1 + test_labels2 
test_images = []
for p in test_labels:
    p = p[:-8]
    p = p+'.npy'
    test_images.append(p)
test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]
test_ds = Dataset(data=test_files, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

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
    model = tiramisu.FCDenseNet_gray_landmark1(n_classes=1).cuda()



model.apply(train_utils.weights_init)
if use_pretrained:
    model = train_utils.load_weights(model, fpath)
loss_CE = nn.CrossEntropyLoss().cuda()
loss_MSE = nn.MSELoss().cuda()
loss_MAE = nn.L1Loss().cuda()
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
    trn_loss = train_utils.train_LandD(writer, model, train_loader,train_ds, optimizer, scaler, loss_MSE, epoch, lambda_2=50.0)

    print('Epoch {:d}\nTrain - Loss: {:.4f} '.format(epoch, trn_loss))
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.cuda.empty_cache()
    
    ### Validation ###
    since = time.time()
    val_loss = train_utils.test_LandD(model,val_loader, optimizer, loss_MSE, lambda_2=50.0)
    scheduler.step(val_loss)
    
    print(f"lr: {optimizer.param_groups[0]['lr']}")
    print('Val Synthetic - Loss: {:.4f} '.format(val_loss ))
    time_elapsed = time.time() - since  
    print('Validation Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    torch.cuda.empty_cache()
    
    ### Inferance ###
    if epoch %5 == 1:
        since = time.time()
        test_loss, test_d1 = train_utils.inferance_LandD(model,test_loader, optimizer, loss_MSE, lambda_2=50.0)
        print('Inferance Synthetic - Loss: {:.4f} - distance: {:.6f}'.format(test_loss ,np.median(test_d1)))
        print(f"distance values for each image in the test loader: {test_d1}")
        time_elapsed = time.time() - since  
        print('Inferance Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        torch.cuda.empty_cache()
    
    train_utils.save_weights(model, epoch, val_loss, WEIGHTS_PATH)
writer.close()


    