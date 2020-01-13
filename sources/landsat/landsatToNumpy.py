# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:09:23 2019

@author: Alexandra
"""
from osgeo import gdal
import numpy as np
import os


def openFile(filepath):
    # Open the file:
    raster = gdal.Open(filepath) 
    
    band = raster.ReadAsArray()
    
    # Check type of the variable 'band'
    #print(type(band))
    
    return band

    
yourpath = r'K:\gis data\LC08_L1TP_161029_20191014_20191018_01_T1'

filepath = []


for file in os.listdir(yourpath):
    if file.endswith(".TIF"):
        filepath.append(os.path.join(yourpath, file))

print(filepath)

raster = gdal.Open(filepath[0])  
    # Dimensions
print(raster.RasterXSize)
print(raster.RasterYSize)
x = raster.RasterXSize
y = raster.RasterYSize

arr = list()
arr.append(openFile(filepath[0]).reshape(y,x))
arr.append(openFile(filepath[1]).reshape(y,x))
arr.append(openFile(filepath[2]).reshape(y,x))
arr.append(openFile(filepath[3]).reshape(y,x))
arr.append(openFile(filepath[4]).reshape(y,x))
arr.append(openFile(filepath[5]).reshape(y,x))
arr.append(openFile(filepath[6]).reshape(y,x))
arr.append(openFile(filepath[7]).reshape(y,x))
arr.append(openFile(filepath[8]).reshape(y,x))
#
arr.append(openFile(filepath[10]).reshape(y,x))
#arr.append(openFile(filepath[11]).reshape(y,x))

#arr.append(openFile(filepath[9]).reshape(x,y))

b = np.array(arr)
b = np.transpose(b, (1, 2, 0))


np.save("landsat", b)

print(b.shape)




