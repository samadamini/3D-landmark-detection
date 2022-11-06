import time
import os
import glob
import numpy as np
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

from monai.data import DataLoader, Dataset
from monai.networks.nets import HighResNet
import nibabel as nib
import monai
from pathlib import Path
from scipy import ndimage
from tqdm.notebook import tqdm
import re 

def distance_mm(c1,c2,res):
    d = ((c1[0]-c2[0])*res[0])**2 + ((c1[1]-c2[1])*res[1])**2 + ((c1[2]-c2[2])*res[2])**2
    return np.sqrt(d)

def data_loader():
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
    ## initial augmentation
    for path in zip([test_data_path],[test_data_raw]):
        if not os.path.exists(path[0]+affix):
            ## Creating an augmentation object
            data_augmentation = data_preprocessing.preprocessing_aug(path[1],image_size=(64, 64, 64),train=False)
            ## generating the augmented images with different cropsize and axes order 
            counter = data_augmentation.generate_image_label(path[0]+affix,sigma=4)
            #counter = data_augmentation.generate_image_label_multi_landmark(num_landmark, path[0],sigma=3)
            print(f"{counter} images was found images were generated and saved in {path[0]+affix}.")



    batch_size = 4
    learning_rate = 1e-4*0.5
    weight_decay = 1e-5
    N_EPOCHS = 200
    torch.cuda.manual_seed(0)

    generat_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_labels = glob.glob(val_data_path+affix+'/*blob.npy')
    val_images = []
    for p in val_labels:
        p = p[:-8]
        p = p+'.npy'
        val_images.append(p)
    val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]
    val_ds = Dataset(data=val_files, transform=generat_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    test_labels1 = glob.glob(test_data_path+'/*blob.npy')
    test_labels2 = glob.glob(val_data_path+affix+'/*200xyzblob.npy')
    test_labels = test_labels1 + test_labels2
    test_images = []
    for p in test_labels:
        p = p[:-8]
        p = p+'.npy'
        test_images.append(p)
    test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]
    test_ds = Dataset(data=test_files, transform=generat_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return val_loader,test_loader


def test_output(val_loader,test_loader,output_folder,model_choice):
    
    assert Model_choice in ["unet1", "unet2", "unet3", "highresnet"], "Choose a model between unet1, unet2, unet3, and highresnet "
    WEIGHTS_PATH = Path('/home/ec2-user/SageMaker/autodetect/pretrained_weights') / f'latest_{Model_choice}.th'
    
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


    model = train_utils.load_weights(model, WEIGHTS_PATH)

    ### Multi-GPU ###
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #,device_ids=[0, 1, 2, 3]
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    model.eval()
    test_loss = 0
    root_dir = os.getcwd()
    output1 = os.path.join(root_dir, output_folder)
    with torch.no_grad():     
        for idx, data in enumerate(val_loader): #### which loader
            inputs = data['image'].cuda() 
            target = data['label'].cuda()
            ### Prediction tensor ### 
            output = model(inputs)
            np.save(os.path.join(output1, f"in{idx}"),inputs.cpu().detach().numpy())
            np.save(os.path.join(output1, f"out{idx}"),output.cpu().detach().numpy())
            np.save(os.path.join(output1, f"target{idx}"),target.cpu().detach().numpy())
        for idx, data in enumerate(test_loader): #### which loader
            inputs = data['image'].cuda() 
            target = data['label'].cuda()
            ### Prediction tensor ### 
            output = model(inputs)
            np.save(os.path.join(output1, f"Tin{idx}"),inputs.cpu().detach().numpy())
            np.save(os.path.join(output1, f"Tout{idx}"),output.cpu().detach().numpy())
            np.save(os.path.join(output1, f"Ttarget{idx}"),target.cpu().detach().numpy())


def distance_test(val_loadert,pre,output_folder):
    root_dir = os.getcwd()
    res = [0.29231285570766]*3
    loss_val = []
    loader = val_loadert
    output = os.path.join(root_dir, output_folder)
    for i in tqdm(range(len(loader))):
        out0 = np.load(os.path.join(output, f"{pre}out{i}.npy"))
        target0 = np.load(os.path.join(output, f"{pre}target{i}.npy"))
        for bn in range(loader.batch_size):
            surr = np.ma.masked_less(out0[bn][0],0.5*np.max(out0[bn][0]))
            c1 = ndimage.measurements.center_of_mass(surr)
            if c1 is np.nan:
                if np.isnan(surr):
                    print("surr have nan")

            surr = np.ma.masked_less(target0[bn][0],0.5*np.max(target0[bn][0]))
            c2 = ndimage.measurements.center_of_mass(surr)
            loss_val.append(distance_mm(c1,c2,res))
    return loss_val

if __name__ == "__main__":
    output_folder = "output_unet2_64" ## Create an empty folder with a name that you put here 
    model_choice = "unet2" ### unet1, unet2, and unet3 are small, medium, and large unet models, respectively.
    
    ## data loader
    val_loader,test_loader = data_loader()
    
    ## Generating output/input/groundtruth for validation set and test set
    test_output(val_loader,test_loader,output_folder,model_choice) ## you can comment this part after generating the output one time
    
    pre = "" ## prefix for the validation data generated by test_output() function
    loss_val = distance_test(val_loader,pre,output_folder)
    pre = "T" ## prefix for the Test data generated by test_output() function
    loss_test = distance_test(test_loader,pre,output_folder)
    print(f"Test loss mean {np.nanmean(loss_test)} and median {np.nanmedian(loss_test)}")
    print(f"Val loss mean {np.nanmean(loss_val)} and median {np.nanmedian(loss_val)}")
    print(f"Test losses are {list(np.round(loss_test, decimals = 2))}")
    print(f"Val losses are {list(np.round(loss_val, decimals = 2))}")
    
