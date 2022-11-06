# autodetect

## Fully cloud based andmark detection in 3D angiography images
![alt text](https://github.com/philips-internal/autodetect/blob/main/Pipeline.png?raw=true)

### Clone the repo and navigate to where the repo is cloned:
`cd ./autodetect`

### To install the dependencies install the requirments.txt
`pip install -r requirements.txt`

## How to start
There are already data in this repo that can be used for either training or inference. Go ahead with `screen -dmsL test1 python3 train.py`

If there is no data, just put the dicom files in the **aneurysm_dicom** folder and the annotation in the  **aneurysm_annotation**. Then run `train_val_test_split.py` to create the **data** folder, resulting in generating train, validation, and test files. Then run the `train.py` which creates the **data_augmented** folder.

### Scripts
`train.py` includes three different *Unet* models with different size and one *Highresnet* model. You can change the **model_choice** parameter in `train.py` to start training any of these 4 models. The augmentation doesn't execute if the folder **data_augmented** contains train_data_augmented, So empty the folder or change the **affix** parameter.

`train_multiLandmark_multiCahnnel.py` is a training script that can be used in a multi landmark/multi output channel scenario. There is no pretrained file for these models. You can change the NN by *model_choice* parameter. The data for train/val/test will be generated and saved in **data_augmented**. 

`inference`: calculates the distance between the prediction and the ground truth by generating the input/output/label and save them into the *output_folder* parameter. 

### Data folder decription
*aneurysm_annotation*: contains the annotation files.
*aneurysm_dicom/aneurysm_dicom*: contains the dicom files.

*data*: contains the splitted data (dicom and annotations) generated by train_val_test_split.py -- It works on dicom files that starts with "Pat" or "3dra".
*data_augmented*: This data was generated after running train.py -- need to change *affix* parameter for generating more/new augmented data. 

*output_unet2_64*: There exist some output results from Unet2 model that can be used to check the performance

*screen_logs*: contains some of the logs after training Unet1, Unet2, and Unet3.

### Tips
Use screen comnmand for training process to the log file, e.g.`screen -dmsL test1 python3 train.py`
Most changes can be done in `utils/training_monai.py` and `scr/data_preprocessing.py`

