# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 23:07:03 2019

@author: Alexandra
"""

from osgeo import gdal
import math

#filepath = r"K:\gis data\aral\LC08_L1TP_161028_20190320_20190326_01_T1\LC08_L1TP_161028_20190320_20190326_01_T1_B10.TIF"
filepath = r"K:\gis data\nextgis\22.12\10\LC08_L1TP_161029_20190507_20190521_01_T1_B10.TIF"

# Open the file:
raster = gdal.Open(filepath)

# Check type of the variable 'raster'
print(type(raster))

# Projection
print(raster.GetProjection())

# Dimensions
print(raster.RasterXSize)
print(raster.RasterYSize)

# Number of bands
print(raster.RasterCount)

# Metadata for the raster dataset
print(raster.GetMetadata())

# Read the raster band as separate variable
band = raster.GetRasterBand(1)

# Check type of the variable 'band'
type(band)

# Data type of the values
gdal.GetDataTypeName(band.DataType)

# Compute statistics if needed
if band.GetMinimum() is None or band.GetMaximum()is None:
    band.ComputeStatistics(0)
    print("Statistics computed.")

# Fetch metadata for the band
band.GetMetadata()

# Print only selected metadata:
print ("[ NO DATA VALUE ] = ", band.GetNoDataValue()) # none
print ("[ MIN ] = ", band.GetMinimum())
print ("[ MAX ] = ", band.GetMaximum())

max_radiance = band.GetMaximum()

RADIANCE_MINIMUM_BAND_10 = 0.10033
RADIANCE_MAXIMUM_BAND_10 = 22.00180
RADIANCE_MULT_BAND_10 = 3.3420e-04
RADIANCE_ADD_BAND_10 = 0.1
K2_CONSTANT_BAND_10 = 1321.0789
K1_CONSTANT_BAND_10 = 774.8853

print(max_radiance)

TOARadiance = RADIANCE_MULT_BAND_10 * max_radiance + RADIANCE_ADD_BAND_10
print("TOARadiance " + str(TOARadiance))

TKelvin = K2_CONSTANT_BAND_10 / math.log(K1_CONSTANT_BAND_10/TOARadiance + 1 , math.e)
TCelsium = TKelvin - 237.15

print(TCelsium)

#62.327563056321566
#56.413779831710855

#79.52773561769433
#125.82067968691663



