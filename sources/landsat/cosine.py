# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:21:28 2020

@author: Alexandra
"""

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

def openFile(filepath):
    # Open the file:
    raster = gdal.Open(filepath) 
    band = raster.ReadAsArray()
    
    # Check type of the variable 'band'
    #print(type(band))
    
    return band

    
dir1 = r'J:\GIS_DATA\Fire\2019\LC08_L1TP_135018_20190704_20190718_01_T1'
name1 = "\LC08_L1TP_135018_20190704_20190718_01_T1_B"
dir2 = r'J:\GIS_DATA\Fire\2019\LC08_L1TP_135018_20190805_20190820_01_T1'
name2 = "\LC08_L1TP_135018_20190805_20190820_01_T1_B"

filepath = dir2+name2+"1.TIF"

raster = gdal.Open(filepath) 
band = raster.ReadAsArray()

print(band) 
plt.imshow(band)
plt.show()
    # Dimensions
print(raster.RasterXSize)
print(raster.RasterYSize)
x = raster.RasterXSize
y = raster.RasterYSize

#result = np.zeros((y,x))
#
#for i in [5,6,7]:
#    filepath1 = dir1+name1+str(i)+".TIF"
#    t1 = openFile(filepath1)
#    arr1 = t1.reshape(y,x)
#    filepath2 = dir2+name2+str(i)+".TIF"
#    t2 = openFile(filepath2)
#    arr2 = t2.reshape(y,x)
#    m = abs(arr1 - arr2)
#    plt.imshow(m)
#    plt.show()
#    result = np.add(result, m)
#
##result = np.sqrt(result)
#
#plt.imshow(result, cmap = 'gray')
#plt.show()


l1 = []
l2 = []
for i in [5,6,7]:
    filepath1 = dir1+name1+str(i)+".TIF"
    t1 = openFile(filepath1)
    arr1 = t1.reshape(y,x)
    l1.append(arr1)
    filepath2 = dir2+name2+str(i)+".TIF"
    t2 = openFile(filepath2)
    arr2 = t2.reshape(y,x)
    l2.append(arr2)
    
a1 = np.asarray(l1)
print(a1.shape)
a2 = np.asarray(l2)
print(a2.shape)

a1 = np.transpose(a1, (1, 2, 0))
a2 = np.transpose(a2, (1, 2, 0))

print(a1[0,1,])

r = np.zeros((y,x))
import pirson

for i in range(0, x):
    for j in range(0, y):
        r[j, i] = pirson.pearson(a1[j,i,], a2[j,i,])
        
#result = np.sqrt(result)

plt.imshow(r, cmap = 'gray')
plt.show()





print(raster.shape)