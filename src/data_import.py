import numpy as np
import os
from skspatial.objects import Plane
from src.SaveStateReader import *
import pydicom
import json

import warnings
warnings.filterwarnings('ignore')
  
class Data:
    ### imports neck annotation, image and their attributes 
    def __init__(self, annotation_path, image_path, RESOLUTION=[]):
        
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.RESOLUTION = RESOLUTION
        
        self.annot = json.load(open(annotation_path))
        
        self.image_base = pydicom.dcmread(image_path)
        self.image = self.image_base.pixel_array
        
        if len(RESOLUTION)<3:
            try:
                xy_res = pydicom.dcmread(image_path).PixelSpacing ### there may be a better way to read spacing info from metadata
                z_res = pydicom.dcmread(image_path).SliceThickness
                if xy_res[0]==xy_res[1] and xy_res[0]==z_res:
                    self.RESOLUTION = [xy_res[0]]*3
                else:
                    self.RESOLUTION = [xy_res[0], xy_res[1], z_res]
            except:
                print('Please input resolution values in mm/px unit as RESOLUTION=[xy_res[0], xy_res[1], z_res].')
                print(self.image_base)
                
        self.neck_coords, self.neck_plane = Data.get_neck_plane(self)
                
    
    def get_neck_coords(self):
        #print(RESOLUTION)    
        #annot = json.load(open(annotation_path))
        labels = [x['label'] for x in self.annot['inVolumeSaveData']['savedMeasurements'] ]

        neck_coords={}        
        if 'Neck Diameter' in labels:
            neck_data = self.annot['inVolumeSaveData']['savedMeasurementsData'][labels.index('Neck Diameter')]['data']

            neck_coords_temp = LoadRingMeasurement(neck_data)

            x_temp=[]; y_temp=[]; z_temp=[]
            for i in range(len(neck_coords_temp)):
                x_temp.append(neck_coords_temp[i].x)
                y_temp.append(neck_coords_temp[i].y)
                z_temp.append(neck_coords_temp[i].z)

            x=[self.image.shape[0] - i/self.RESOLUTION[0] for i in x_temp]; y=[i/self.RESOLUTION[1] for i in y_temp]; z=[i/self.RESOLUTION[2] for i in z_temp]

            neck_coords = {'x': x, 'y': y, 'z': z}
            
        else:
            print("Aneurysm neck is not labeled.")
        
        return neck_coords
    
    def get_neck_plane(self):

        neck_coords = Data.get_neck_coords(self)

        neck_coords_list=[]
        for i in range(len(neck_coords['x'])):
            neck_coords_list.append([ neck_coords['x'][i], neck_coords['y'][i], neck_coords['z'][i] ])

        neck_plane = Plane.best_fit(np.array(neck_coords_list))

        return neck_coords, neck_plane