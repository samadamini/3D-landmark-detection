from src import data_import
import os
import glob
import pandas as pd
from ast import literal_eval
import pickle5 as pickle
import numpy as np
import itertools
from tqdm import tqdm
import pydicom ## in case skspatial is not found try "pip install scikit-spatial"

class preprocessing_aug():
    """Preprocessing of neurysm Landmarks dataset."""

    def __init__(self, data_dir, data_aug_dir=None, *, image_size=(128, 128, 128),train=True):
        """
        Args:
            data_dir (string): Path to the images and annotation in .dcm and .state format
        """
        self.data_dir = data_dir
        self.data_aug_dir = data_aug_dir

        self.image_files = glob.glob(self.data_dir + '/*.dcm')
        self.annotation_files = glob.glob(self.data_dir + '/*.state')
        if data_aug_dir is not None:
            self.image_files_aug = glob.glob(self.data_aug_dir + '/*.npy')

        self.img_size = image_size
        self.train = train

    def file_list(path,endwith = None):
        '''
        returns a list
        path: a path to list all the files ending with endwith parameters
        '''
        list_of_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if endwith == None:
                    if not file.endswith('.DS_Store'):
                        list_of_files.append(os.path.join(root,file))
                elif file.endswith(endwith):
                    list_of_files.append(os.path.join(root,file))
                else:
                    pass
        return list_of_files

    def annotation_finder(self,image_path):
        file = image_path.split('/')[-1]
        annotation_path = 0
        if file.startswith("Pat") | file.startswith("3dra"):
            for name in self.annotation_files:
                if name.split('/')[-1].split(' ')[0] == file[:-4]:
                    annotation_path = name
                    #print(f"The annotation file name {annotation_path.split('/')[-1]} was found for image file name {image_path.split('/')[-1]}")
                    break
        else:
            print("The name of image path starts with something unknown!")
        if annotation_path == 0:
            print(f"The name of image {image_path} was not found in the Annotation file directory!")
        return annotation_path

    def resize_data(self,data):
        initial_size_x = data.shape[0]
        initial_size_y = data.shape[1]
        initial_size_z = data.shape[2]

        new_size_x = self.img_size[0]
        new_size_y = self.img_size[1]
        new_size_z = self.img_size[2]

        delta_x = initial_size_x / new_size_x
        delta_y = initial_size_y / new_size_y
        delta_z = initial_size_z / new_size_z

        new_data = np.zeros((new_size_x, new_size_y, new_size_z))

        for x, y, z in itertools.product(range(new_size_x),
                                         range(new_size_y),
                                         range(new_size_z)):
            new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

        return new_data

    def axes_swapping(self,img1, axes_order):
        assert axes_order in ['xzy','yxz','zyx','zxy','yzx'], "axes order should be in 'xzy','yxz','zyx','zxy','yzx']"
        if axes_order == 'xzy':
            img1_swapped = np.swapaxes(img1, 1, 2)
        elif axes_order == 'yxz':
            img1_swapped = np.swapaxes(img1, 0, 1)
        elif axes_order == 'zyx':
            img1_swapped = np.swapaxes(img1, 0, 2)
        elif axes_order == 'zxy':
            img1_swapped = np.swapaxes(img1, 0, 1)
            img1_swapped = np.swapaxes(img1_swapped, 0, 2)
        else:
            img1_swapped = np.swapaxes(img1, 0, 1)
            img1_swapped = np.swapaxes(img1_swapped, 1, 2)
        
        return img1_swapped

    def generate_heatmap_from_locations(self, x, y, z,im_size,sigma = 4.0):
        
        heatmap = np.zeros((im_size[0], im_size[1],im_size[2]), dtype=np.double)

        if y is not None:
            sigma_2 = sigma ** 2

            x_grid, y_grid, z_grid = np.meshgrid(np.arange(im_size[0]),np.arange(im_size[1]), np.arange(im_size[2]), sparse=False, indexing='ij')
            source_grid_x = x_grid - x
            source_grid_y = y_grid - y
            source_grid_z = z_grid - z

            heatmap = np.exp(-(source_grid_x ** 2 + source_grid_y ** 2 + source_grid_z ** 2) / (2.0 * sigma_2))
            heatmap = np.nan_to_num(heatmap, 0.0, 0.0, 0.0)

        return heatmap
    def cropping(self,center,CRPP,img_c):
        if CRPP>150:
            return img_c
        c0, c1, c2 = center[0], center[1], center[2]
        x0, x1, y0, y1, z0, z1 = c0-CRPP, c0+CRPP, c1-CRPP, c1+CRPP, c2-CRPP, c2+CRPP
        p00, p01, p10, p11, p20, p21 = np.zeros(6, dtype=int)
        if x0<0:
            p00 = np.abs(x0)
        if y0<0:
            p10 = np.abs(y0)
        if z0<0:
            p20 = np.abs(z0)
        if x1>img_c.shape[0]:
            p01 = x1 - img_c.shape[0]
        if y1>img_c.shape[1]:
            p11 = y1 - img_c.shape[1]
        if z1>img_c.shape[2]:
            p21 = z1 - img_c.shape[2]
        img_c = np.pad(img_c, ((p00,p01), (p10,p11), (p20, p21)), 'constant')
        image_temp_c = np.zeros((CRPP*2,CRPP*2,CRPP*2))
        x0, x1, y0, y1, z0, z1 = x0+p00, x1+p00, y0+p10, y1+p10, z0+p20, z1+p20 
        image_temp_c[:,:,:] = img_c[x0:x1, y0:y1, z0:z1]
        return image_temp_c
    def shift_center(self,center,CRPP):
        if CRPP>150:
            return center
        divider = 1
        while True:
            if np.all(np.abs((np.array((192,148,192))-np.array(center))//divider) < np.array([CRPP,CRPP,CRPP])//2):
                break
            else:
                divider += 1
        return (np.array((192,148,192))-np.array(center))//divider + np.array(center)

    
    def generate_image_label(self,dir_tosave = 'img_label', *,crop_size=[60,80,100,120], ax_orders=['xzy','yxz','zyx','zxy','yzx'],sigma=4.0, resolution=[0.29231285570766]*3):
        counter = 0   
        for i in tqdm(range(len(self.image_files)), position=0, leave=True):
            image_path_target = self.image_files[i]
            annotation_path_target = self.annotation_finder(image_path_target) ## finding the corresponding annotation file
            if annotation_path_target != 0:
                data = data_import.Data(annotation_path_target, image_path_target, resolution) # reading the image/annotation

                neck_coords = data.neck_coords
                neck_plane = data.neck_plane

                xc,yc,zc = neck_plane.point
                center = [int(xc), int(yc), int(zc)]

                ### Volume:
                image = pydicom.dcmread(image_path_target)
                img = image.pixel_array
                ## Normalization
                img = img-img.min()
                img = img/img.max()
                counter += 1
                ### heatmap
                heatmap = self.generate_heatmap_from_locations(center[0], center[1], center[2], img.shape,sigma)

                if self.train:
                    for CRP in crop_size:
                        ### Cropping
                        center_shifted = self.shift_center(center,CRP)
                        image_temp = self.cropping(center_shifted,CRP,img)
                        heatmap_temp = self.cropping(center_shifted,CRP,heatmap)


                        #Rx, Ry, Rz = self.img_size[0]/image_temp.shape[0], self.img_size[1]/image_temp.shape[1], self.img_size[2]/image_temp.shape[2]

                        ### Resizing 
                        img_resized = self.resize_data(image_temp)
                        heatmap_resized = self.resize_data(heatmap_temp)

                        ### Saving the images
                        newpath_data = dir_tosave
                        if not os.path.exists(newpath_data):
                            os.makedirs(newpath_data) 

                        img_name = image_path_target.split('/')[-1][:-4]

                        np.save(newpath_data+'/'+img_name+'_'+str(CRP)+'xyz',img_resized)
                        np.save(newpath_data+'/'+img_name+'_'+str(CRP)+'xyz'+'blob',heatmap_resized)


                        for ax_order in ax_orders:  
                            np.save(newpath_data+'/'+img_name+'_'+str(CRP)+ax_order,self.axes_swapping(img_resized,ax_order))
                            np.save(newpath_data+'/'+img_name+'_'+str(CRP)+ax_order+'blob',self.axes_swapping(heatmap_resized,ax_order))

                else:
                    ### resize the heatmap
                    #Rx, Ry, Rz = self.img_size[0]/img.shape[0], self.img_size[1]/img.shape[1], self.img_size[2]/img.shape[2]
                    heatmap = self.generate_heatmap_from_locations(center[0], center[1], center[2], img.shape,sigma)

                    heatmap_resized = self.resize_data(heatmap)
                    img_resized = self.resize_data(img)
                    ### Saving the images
                    newpath_data = dir_tosave 
                    if not os.path.exists(newpath_data):
                        os.makedirs(newpath_data) 

                    img_name = image_path_target.split('/')[-1][:-4]

                    np.save(newpath_data+'/'+img_name,img_resized)
                    np.save(newpath_data+'/'+img_name+'blob',heatmap_resized)


            else:
                print(f'{image_path_target} was not found')
        return counter
    
    
    def generate_image_label_multi_landmark(self,num_landmarks = 4,dir_tosave = 'img_label', *,crop_size=[60,80,100,120], ax_orders=['xzy','yxz','zyx','zxy','yzx'],sigma=4.0, resolution=[0.29231285570766]*3):
        counter = 0   
        for i in tqdm(range(len(self.image_files)), position=0, leave=True):
            image_path_target = self.image_files[i]
            annotation_path_target = self.annotation_finder(image_path_target) ## finding the corresponding annotation file
            data = data_import.Data(annotation_path_target, image_path_target, resolution) # reading the image/annotation

            neck_coords = data.neck_coords
            neck_plane = data.neck_plane

            xc,yc,zc = neck_plane.point
            center = [int(xc), int(yc), int(zc)]

            ### Volume:
            image = pydicom.dcmread(image_path_target)
            img = image.pixel_array
            ## Normalization
            img = img-img.min()
            img = img/img.max()
            counter += 1
            ### heatmap
            x = np.array(neck_coords['x'])
            y = np.array(neck_coords['y'])
            z = np.array(neck_coords['z'])
            
            n = len(neck_coords['x'])//num_landmarks
            heatmaps = np.zeros((num_landmarks,img.shape[0],img.shape[1],img.shape[2]))
            for LM in range(num_landmarks):
                heatmaps[LM,:,:,:] = self.generate_heatmap_from_locations(np.array([x[n*LM]]).astype('int'), np.array([y[n*LM]]).astype('int'), np.array([z[n*LM]]).astype('int'), img.shape,sigma)

            if self.train:
                for CRP in crop_size:
                    ### Cropping
                    center_shifted = self.shift_center(center,CRP)
                    image_temp = self.cropping(center_shifted,CRP,img)
                    if CRP < 150:
                        heatmaps_cropped = np.zeros((num_landmarks,CRP*2,CRP*2,CRP*2))
                        for LM in range(num_landmarks):
                            heatmaps_cropped[LM,:,:,:] = self.cropping(center_shifted,CRP,heatmaps[LM,:,:,:].squeeze())
                    else:
                        heatmaps_cropped = heatmaps
                    ### Resizing 
                    img_resized = self.resize_data(image_temp)
                    heatmaps_resized = np.zeros((num_landmarks,self.img_size[0],self.img_size[1],self.img_size[2]))
                    for LM in range(num_landmarks):
                        heatmaps_resized[LM,:,:,:] = self.resize_data(heatmaps_cropped[LM,:,:,:].squeeze())
                    ### Saving the images
                    newpath_data = dir_tosave
                    if not os.path.exists(newpath_data):
                        os.makedirs(newpath_data) 

                    img_name = image_path_target.split('/')[-1][:-4]

                    np.save(newpath_data+'/'+img_name+'_'+str(CRP)+'xyz',img_resized)
                    #np.save(newpath_data+'/'+img_name+'_'+str(CRP)+'xyz'+'blob',heatmaps_resized.sum(axis=0))
                    for LM in range(num_landmarks):
                        np.save(newpath_data+'/'+img_name+'_'+str(CRP)+'xyz'+str(LM)+'blob',heatmaps_resized[LM])


                    for ax_order in ax_orders:  
                        np.save(newpath_data+'/'+img_name+'_'+str(CRP)+ax_order,self.axes_swapping(img_resized,ax_order))
                        heatmaps_swapped = np.zeros((num_landmarks,self.img_size[0],self.img_size[1],self.img_size[2]))
                        for LM in range(num_landmarks):
                            heatmaps_swapped[LM,:,:,:] = self.axes_swapping(heatmaps_resized[LM,:,:,:].squeeze(),ax_order)
                            np.save(newpath_data+'/'+img_name+'_'+str(CRP)+ax_order+str(LM)+'blob',heatmaps_swapped[LM])

            else:
                ### resize the heatmap
                heatmaps_resized = np.zeros((num_landmarks,self.img_size[0],self.img_size[1],self.img_size[2]))
                for LM in range(num_landmarks):
                    heatmaps_resized[LM,:,:,:] = self.resize_data(heatmaps[LM,:,:,:].squeeze())
                img_resized = self.resize_data(img)
                ### Saving the images
                newpath_data = dir_tosave 
                if not os.path.exists(newpath_data):
                    os.makedirs(newpath_data) 

                img_name = image_path_target.split('/')[-1][:-4]

                np.save(newpath_data+'/'+img_name,img_resized)
                for LM in range(num_landmarks):
                    np.save(newpath_data+'/'+img_name+str(LM)+'blob',heatmaps_resized[LM])

        return counter
    
    def generate_image_label_from_tf_augmented(self, annotations_path, dir_tosave = 'img_label', *,crop_size=[60,80,100,120], ax_orders=['xzy','yxz','zyx','zxy','yzx'],sigma=4.0, resolution=[0.29231285570766]*3):
        counter = 0   
        df = pd.read_csv(annotations_path)
        for i in tqdm(range(len(self.image_files_aug)), position=0, leave=True):
            image_path_target = self.image_files_aug[i]
            center = df[df.filename==image_path_target.split("/")[-1]].neck_center.values[0]
            center = literal_eval(center)

            ### Volume:
            img = np.load(image_path_target)
            ## Normalization
            img = img-img.min()
            img = img/img.max()
            counter += 1
            ### heatmap
            heatmap = self.generate_heatmap_from_locations(center[0], center[1], center[2], img.shape,sigma)

            if self.train:
                for CRP in crop_size:
                    ### Cropping
                    center_shifted = self.shift_center(center,CRP)
                    image_temp = self.cropping(center_shifted,CRP,img)
                    heatmap_temp = self.cropping(center_shifted,CRP,heatmap)


                    #Rx, Ry, Rz = self.img_size[0]/image_temp.shape[0], self.img_size[1]/image_temp.shape[1], self.img_size[2]/image_temp.shape[2]

                    ### Resizing 
                    img_resized = self.resize_data(image_temp)
                    heatmap_resized = self.resize_data(heatmap_temp)

                    ### Saving the images
                    newpath_data = dir_tosave
                    if not os.path.exists(newpath_data):
                        os.makedirs(newpath_data) 

                    img_name = image_path_target.split('/')[-1][:-4]

                    np.save(newpath_data+'/'+img_name+'_'+str(CRP)+'xyz'+'aug',img_resized)
                    np.save(newpath_data+'/'+img_name+'_'+str(CRP)+'xyz'+'aug'+'blob',heatmap_resized)


                    for ax_order in ax_orders:  
                        np.save(newpath_data+'/'+img_name+'_'+str(CRP)+ax_order+'aug',self.axes_swapping(img_resized,ax_order))
                        np.save(newpath_data+'/'+img_name+'_'+str(CRP)+ax_order+'aug'+'blob',self.axes_swapping(heatmap_resized,ax_order))

            else:
                ### resize the heatmap
                heatmap = self.generate_heatmap_from_locations(center[0], center[1], center[2], img.shape,sigma)
                heatmap_resized = self.resize_data(heatmap)
                img_resized = self.resize_data(img)
                ### Saving the images
                newpath_data = dir_tosave 
                if not os.path.exists(newpath_data):
                    os.makedirs(newpath_data) 

                img_name = image_path_target.split('/')[-1][:-4]

                np.save(newpath_data+'/'+img_name+'aug',img_resized)
                np.save(newpath_data+'/'+img_name+'aug'+'blob',heatmap_resized)

        return counter
    
                    
                    
                    

