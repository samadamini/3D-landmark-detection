import json
import struct
import pydicom
import numpy as np

class Vector3:
    x = 0.0
    y = 0.0
    z = 0.0

class Quaternion:
    x = 0.0
    y = 0.0
    z = 0.0
    w = 0.0

class GradientColorKey:
    r = 0.0
    g = 0.0
    b = 0.0
    a = 0.0
    time = 0.0

class GradientAlphaKey:
    a = 0.0
    time = 0.0

class Color:
    r = 0.0
    g = 0.0
    b = 0.0
    a = 0.0


def LoadLinearMeasurement(data):
    currentReadIndex = 0
    atleast1PointInVolume = bool.from_bytes(data[currentReadIndex:currentReadIndex+1], "little")
    # print(atleast1PointInVolume)
    currentReadIndex += 1

    #localPositioning
    localPosition = Vector3()
    [localPosition.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localPosition.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localPosition.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(localPosition.x)
    # print(localPosition.y)
    # print(localPosition.z)

    localRotation = Quaternion()
    [localRotation.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.w] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(localRotation.x)
    # print(localRotation.y)
    # print(localRotation.z)
    # print(localRotation.w)

    localScale = Vector3()
    [localScale.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localScale.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localScale.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(localScale.x)
    # print(localScale.y)
    # print(localScale.z)

    anchorPosition = Vector3()
    [anchorPosition.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [anchorPosition.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [anchorPosition.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(anchorPosition.x)
    # print(anchorPosition.y)
    # print(anchorPosition.z)

    pivotPosition = Vector3()
    [pivotPosition.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [pivotPosition.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [pivotPosition.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(pivotPosition.x)
    # print(pivotPosition.y)
    # print(pivotPosition.z)

    anchorColor = Color()
    [anchorColor.r] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [anchorColor.g] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [anchorColor.b] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [anchorColor.a] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    pivotColor = Color()
    [pivotColor.r] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [pivotColor.g] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [pivotColor.b] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [pivotColor.a] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    #apparently I don't write down the number of scales so just iterate through the rest
    worldScaleSetterScales = {}
    worldScaleIndex = 0
    while (currentReadIndex < len(data)):
        worldScaleSetterScales[worldScaleIndex] = Vector3()
        [worldScaleSetterScales[worldScaleIndex].x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [worldScaleSetterScales[worldScaleIndex].y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [worldScaleSetterScales[worldScaleIndex].z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        worldScaleIndex +=1

    # print("linear")
    
    return localPosition, localRotation, localScale, anchorPosition, pivotPosition
    
    

def LoadRingMeasurement(data):
    currentReadIndex = 0
    snappingMode = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
    #print(snappingMode)
    currentReadIndex += 4
    showMaxDiameter = bool.from_bytes(data[currentReadIndex:currentReadIndex+1], "little")
    currentReadIndex += 1
    #print(showMaxDiameter)
    atleast1PointInVolume = bool.from_bytes(data[currentReadIndex:currentReadIndex+1], "little")
    #print(atleast1PointInVolume)
    currentReadIndex += 1
    [radius] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    currentMaxRadius = bool.from_bytes(data[currentReadIndex:currentReadIndex+1], "little")
    currentReadIndex += 1

    #localPositioning
    localPosition = Vector3()
    [localPosition.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localPosition.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localPosition.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    #print(localPosition.x)
    #print(localPosition.y)
    #print(localPosition.z)

    localRotation = Quaternion()
    [localRotation.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.w] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    #print(localRotation.x)
    #print(localRotation.y)
    #print(localRotation.z)
    #print(localRotation.w)

    localScale = Vector3()
    [localScale.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localScale.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localScale.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    #print(localScale.x)
    #print(localScale.y)
    #print(localScale.z)

    point0 = Vector3()
    [point0.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [point0.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [point0.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    point1 = Vector3()
    [point1.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [point1.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [point1.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    point2 = Vector3()
    [point2.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [point2.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [point2.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    #points on the ring
    linePositionCount = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
    currentReadIndex += 4
    #print(linePositionCount)

    linePositions = {}
    for i in range(linePositionCount):
        linePositions[i] = Vector3()
        [linePositions[i].z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [linePositions[i].x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [linePositions[i].y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4

    numberOfGradientColorKeys = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
    currentReadIndex += 4
    #print(numberOfGradientColorKeys)

    gradientColorKeys = {}
    for i in range(numberOfGradientColorKeys):
        gradientColorKeys[i] = GradientColorKey()
        [gradientColorKeys[i].r] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [gradientColorKeys[i].g] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [gradientColorKeys[i].b] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [gradientColorKeys[i].a] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [gradientColorKeys[i].time] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
    
    numberOfGradientAlphaKeys = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
    currentReadIndex += 4
    #print(numberOfGradientAlphaKeys)

    gradientAlphaKeys = {}
    for i in range(numberOfGradientAlphaKeys):
        gradientAlphaKeys[i] = GradientAlphaKey()
        [gradientAlphaKeys[i].a] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        [gradientAlphaKeys[i].time] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4

    #print("ring")
    
    return linePositions

def LoadVolumeMeasurement(data):
    currentReadIndex = 0
    wireframeMode = bool.from_bytes(data[currentReadIndex:currentReadIndex+1], "little")
    currentReadIndex += 1  
    currentMaterialIndex = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
    currentReadIndex += 4

    #localPositioning
    localPosition = Vector3()
    [localPosition.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localPosition.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localPosition.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(localPosition.x)
    # print(localPosition.y)
    # print(localPosition.z)

    localRotation = Quaternion()
    [localRotation.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localRotation.w] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

#     print(localRotation.z)
#     print(localRotation.x)
#     print(localRotation.y)
#     print(localRotation.w)

    localScale = Vector3()
    [localScale.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localScale.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [localScale.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(localScale.x)
    # print(localScale.y)
    # print(localScale.z)

    MeshRendererLocalPosition = Vector3()
    [MeshRendererLocalPosition.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [MeshRendererLocalPosition.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [MeshRendererLocalPosition.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(MeshRendererLocalPosition.x)
    # print(MeshRendererLocalPosition.y)
    # print(MeshRendererLocalPosition.z)

    MeshRendererLocalRotation = Quaternion()
    [MeshRendererLocalRotation.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [MeshRendererLocalRotation.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [MeshRendererLocalRotation.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [MeshRendererLocalRotation.w] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(MeshRendererLocalRotation.x)
    # print(MeshRendererLocalRotation.y)
    # print(MeshRendererLocalRotation.z)
    # print(MeshRendererLocalRotation.w)

    MeshRendererLocalScale = Vector3()
    [MeshRendererLocalScale.z] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [MeshRendererLocalScale.x] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4
    [MeshRendererLocalScale.y] = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
    currentReadIndex += 4

    # print(MeshRendererLocalScale.x)
    # print(MeshRendererLocalScale.y)
    # print(MeshRendererLocalScale.z)

    numberOfVertices = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
    currentReadIndex += 4

    vertices = {}
    for i in range(numberOfVertices):
        vertices[i] = Vector3()
        vertices[i].z = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        vertices[i].x = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4
        vertices[i].y = struct.unpack('<f', bytes(data[currentReadIndex:currentReadIndex+4]))
        currentReadIndex += 4

    numberOfTriangles = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
    currentReadIndex += 4

    triangles = {}
    for i in range(numberOfTriangles):
        triangles[i] = int.from_bytes(data[currentReadIndex:currentReadIndex+4], "little")
        currentReadIndex += 4

    # print("volume")
    
    return vertices


def annotation_coordinates(annotation_path, image_path, RESOLUTION=[], height=True, width=True, neck=True, prox_vessel=True, dist_vessel=True, volume=True):
    if len(RESOLUTION)<3:
        try:
            xy_res = pydicom.dcmread(image_path).PixelSpacing
            z_res = pydicom.dcmread(image_path).SliceThickness
            if xy_res[0]==xy_res[1] and xy_res[0]==z_res:
                RESOLUTION = [xy_res[0]]*3
            else:
                RESOLUTION = [xy_res[0], xy_res[1], z_res]
        except:
            print('Please input resolution values in mm/px unit as RESOLUTION=[xy_res[0], xy_res[1], z_res].')
        
    image_base = pydicom.dcmread(image_path)
    image = image_base.pixel_array

    annot = json.load(open(annotation_path))
    labels = [x['label'] for x in annot['inVolumeSaveData']['savedMeasurements'] ]

    height_coords={}
    if height:
        if 'Height' in labels:
            height_data = annot['inVolumeSaveData']['savedMeasurementsData'][labels.index('Height')]['data']

            localPosition, localRotation, localScale, anchorPosition, pivotPosition = LoadLinearMeasurement(height_data)

            x_temp = pivotPosition.x; y_temp = pivotPosition.y; z_temp = pivotPosition.z
            x1=image.shape[0]-x_temp/RESOLUTION[0] ; y1=y_temp/RESOLUTION[1]; z1=z_temp/RESOLUTION[2]

            x_temp = anchorPosition.x; y_temp = anchorPosition.y; z_temp = anchorPosition.z
            x2=image.shape[0]-x_temp/RESOLUTION[0] ; y2=y_temp/RESOLUTION[1]; z2=z_temp/RESOLUTION[2]

            height_coords = {'x': [x1, x2], 'y': [y1, y2], 'z': [z1, z2]}

    width_coords={}        
    if width:
        if 'Width' in labels:
            width_data = annot['inVolumeSaveData']['savedMeasurementsData'][labels.index('Width')]['data']

            width_coords_temp = LoadRingMeasurement(width_data)

            x_temp=[]; y_temp=[]; z_temp=[]
            for i in range(len(width_coords_temp)):
                x_temp.append(width_coords_temp[i].x)
                y_temp.append(width_coords_temp[i].y)
                z_temp.append(width_coords_temp[i].z)

            x=[image.shape[0] - i/RESOLUTION[0] for i in x_temp]; y=[i/RESOLUTION[1] for i in y_temp]; z=[i/RESOLUTION[2] for i in z_temp]

            width_coords = {'x': x, 'y': y, 'z': z}
            
    neck_coords={}        
    if neck:
        if 'Neck Diameter' in labels:
            neck_data = annot['inVolumeSaveData']['savedMeasurementsData'][labels.index('Neck Diameter')]['data']

            neck_coords_temp = LoadRingMeasurement(neck_data)

            x_temp=[]; y_temp=[]; z_temp=[]
            for i in range(len(neck_coords_temp)):
                x_temp.append(neck_coords_temp[i].x)
                y_temp.append(neck_coords_temp[i].y)
                z_temp.append(neck_coords_temp[i].z)

            x=[image.shape[0] - i/RESOLUTION[0] for i in x_temp]; y=[i/RESOLUTION[1] for i in y_temp]; z=[i/RESOLUTION[2] for i in z_temp]

            neck_coords = {'x': x, 'y': y, 'z': z}

    prox_vessel_coords={}
    if prox_vessel:
        if 'Proximal Vessel Diameter' in labels:
            prox_vessel_data = annot['inVolumeSaveData']['savedMeasurementsData'][labels.index('Proximal Vessel Diameter')]['data']

            prox_vessel_coords_temp = LoadRingMeasurement(prox_vessel_data)

            x_temp=[]; y_temp=[]; z_temp=[]
            for i in range(len(prox_vessel_coords_temp)):
                x_temp.append(prox_vessel_coords_temp[i].x)
                y_temp.append(prox_vessel_coords_temp[i].y)
                z_temp.append(prox_vessel_coords_temp[i].z)

            x=[image.shape[0] - i/RESOLUTION[0] for i in x_temp]; y=[i/RESOLUTION[1] for i in y_temp]; z=[i/RESOLUTION[2] for i in z_temp]

            prox_vessel_coords = {'x': x, 'y': y, 'z': z}

    dist_vessel_coords={}
    if dist_vessel:
        if 'Distal Vessel Diameter' in labels:
            dist_vessel_data = annot['inVolumeSaveData']['savedMeasurementsData'][labels.index('Distal Vessel Diameter')]['data']

            dist_vessel_coords_temp = LoadRingMeasurement(dist_vessel_data)

            x_temp=[]; y_temp=[]; z_temp=[]
            for i in range(len(dist_vessel_coords_temp)):
                x_temp.append(dist_vessel_coords_temp[i].x)
                y_temp.append(dist_vessel_coords_temp[i].y)
                z_temp.append(dist_vessel_coords_temp[i].z)

            x=[image.shape[0] - i/RESOLUTION[0] for i in x_temp]; y=[i/RESOLUTION[1] for i in y_temp]; z=[i/RESOLUTION[2] for i in z_temp]

            dist_vessel_coords = {'x': x, 'y': y, 'z': z}

    volume_mesh_coords = {}
    if volume:
        if 'Volume' in labels:
            volume_data = annot['inVolumeSaveData']['savedMeasurementsData'][labels.index('Volume')]['data']

            vertices = LoadVolumeMeasurement(volume_data)
            x_temp=[]; y_temp=[]; z_temp=[]
            for i in range(len(vertices)):
                x_temp.append(vertices[i].x[0])
                y_temp.append(vertices[i].y[0])
                z_temp.append(vertices[i].z[0])

            x=[image.shape[0] - i/RESOLUTION[0] for i in x_temp]; y=[i/RESOLUTION[1] for i in y_temp]; z=[i/RESOLUTION[2] for i in z_temp]
            volume_mesh_coords = {'x': x, 'y': y, 'z': z}

    return height_coords, width_coords, neck_coords, prox_vessel_coords, dist_vessel_coords, volume_mesh_coords
    

def annotation_measurements(annotation_path, height=True, width=True, neck=True, prox_vessel=True, dist_vessel=True, volume=True):
    
    height='NA'; width='NA'; neck='NA'; prox_vessel='NA'; dist_vessel='NA'; volume='NA'
    annot = json.load(open(annotation_path))
    labels = [x['label'] for x in annot['inVolumeSaveData']['savedMeasurements'] ]

    if height:
        if 'Height' in labels:
            height = annot['inVolumeSaveData']['savedMeasurements'][labels.index('Height')]['measuredValue']

    if width:
        if 'Width' in labels:
            width = annot['inVolumeSaveData']['savedMeasurements'][labels.index('Width')]['measuredValue']
            
    if neck:
        if 'Neck Diameter' in labels:
            neck = annot['inVolumeSaveData']['savedMeasurements'][labels.index('Neck Diameter')]['measuredValue']

    if prox_vessel:
        if 'Proximal Vessel Diameter' in labels:
            prox_vessel = annot['inVolumeSaveData']['savedMeasurements'][labels.index('Proximal Vessel Diameter')]['measuredValue']

    if dist_vessel:
        if 'Distal Vessel Diameter' in labels:
            dist_vessel = annot['inVolumeSaveData']['savedMeasurements'][labels.index('Distal Vessel Diameter')]['measuredValue']

    if volume:
        if 'Volume' in labels:
            volume = annot['inVolumeSaveData']['savedMeasurements'][labels.index('Volume')]['measuredValue']

    return height, width, neck, prox_vessel, dist_vessel, volume



'''
#basic loader
filePath =  input('Input your path:').replace('"', (''))
with open(filePath) as file_handle:
    data = json.load(file_handle)

    index = 0
    for inVolumeSavedMeasurement in data['inVolumeSaveData']['savedMeasurements']:
        print("reading " + inVolumeSavedMeasurement['label'])
        tooltype = int(inVolumeSavedMeasurement['measurementToolType'])
        if (tooltype == 1):
            LoadLinearMeasurement(data['inVolumeSaveData']["savedMeasurementsData"][index]['data'])
        elif (tooltype == 2):
            LoadRingMeasurement(data['inVolumeSaveData']["savedMeasurementsData"][index]['data'])
        elif (tooltype == 3):
            LoadVolumeMeasurement(data['inVolumeSaveData']["savedMeasurementsData"][index]['data'])
        index += 1
'''