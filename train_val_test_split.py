import splitfolders
import glob
import os
import shutil

def annotation_finder(image_path,ann_list):
    file = image_path.split('/')[-1]
    annotation_path = 0
    if file.startswith("Pat") | file.startswith("3dra"):
        for name in ann_list:
            if name.split('/')[-1].split(' ')[0] == file[:-4]:
                annotation_path = name
                #print(f"The annotation file name {annotation_path.split('/')[-1]} was found for image file name {image_path.split('/')[-1]}")
                break
    else:
        print("The name of image path starts with something unknown!")
    if annotation_path == 0:
        print(f"The name of image {image_path} was not found in the Annotation file directory!")
    return annotation_path

current_path = os.getcwd()
print("split folder is running")
splitfolders.ratio(current_path+"aneurysm_dicom", output="data", seed=42, ratio=(.8, .1, .1), group_prefix=None, move=False)
print("split folder done")

image_train = glob.glob('data/train/aneurysm_dicom' + '/*.dcm')
image_val = glob.glob('data/val/aneurysm_dicom' + '/*.dcm')
image_test = glob.glob('data/test/aneurysm_dicom' + '/*.dcm')
annotation_files = glob.glob('aneurysm_annotation' + '/*.state')

for f in image_train:
    ann_train_name = annotation_finder(f,annotation_files)
    if os.path.isfile(ann_train_name):
        shutil.copy(ann_train_name, 'data/train/aneurysm_dicom')

for f in image_val:
    ann_val_name = annotation_finder(f,annotation_files)
    if os.path.isfile(ann_val_name):
        shutil.copy(ann_val_name, 'data/val/aneurysm_dicom')
    
for f in image_test:
    ann_test_name = annotation_finder(f,annotation_files)
    if os.path.isfile(ann_test_name):
        shutil.copy(ann_test_name, 'data/test/aneurysm_dicom')
    