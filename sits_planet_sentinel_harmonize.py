# coding: utf-8

# Raster scatter plot
# 
# Compare two raster dataset, plot scatter plot, harmonize data
# Krištof Oštir
# UL FGG
# (c) 2020

# %%
# Imports
import sys
import matplotlib.pyplot as plt
import gdal

# %%
import satimagets as sits


# %%
# Images
# # PlanetScope
# file_1 = "D:/GeoData/PlanetCalibration/PlanetScope/20170720_090825_1032/20170720_090825_1032_3B_AnalyticMS.tif"
# file_1_udm = "D:/GeoData/PlanetCalibration/PlanetScope/20170720_090825_1032/20170720_090825_1032_3B_AnalyticMS_DN_udm.tif"
# file_1_ndvi = "D:/GeoData/PlanetCalibration/PlanetScope/20170720_090825_1032/20170720_090825_1032_3B_AnalyticMS_NDVI.tif"
# # Sentinel-2
# file_2 = "D:/GeoData/PlanetCalibration/Sentinel-2/L1C_T33TVM_A010844_20170720T100027_con/RT_T33TVM_A010844_20170720T100027_stack.tif"
# file_2_ndvi = "D:/GeoData/PlanetCalibration/Sentinel-2/L1C_T33TVM_A010844_20170720T100027_con/RT_T33TVM_A010844_20170720T100027_stack_NDVI.tif"
# Subset
# PlanetScope
file_1 = "D:/GeoData/PlanetCalibration/PlanetScope/20170720_090825_1032/20170720_090825_1032_3B_AnalyticMS_sub.tif"
file_1_udm = "D:/GeoData/PlanetCalibration/PlanetScope/20170720_090825_1032/20170720_090825_1032_3B_AnalyticMS_DN_udm_sub.tif"
file_1_ndvi = "D:/GeoData/PlanetCalibration/PlanetScope/20170720_090825_1032/20170720_090825_1032_3B_AnalyticMS_NDVI_sub.tif"
# Sentinel-2
file_2 = "D:/GeoData/PlanetCalibration/Sentinel-2/L1C_T33TVM_A010844_20170720T100027/T33TVM_A010844_20170720T100027_stack_sub.tif"
file_2_ndvi = "D:/GeoData/PlanetCalibration/Sentinel-2/L1C_T33TVM_A010844_20170720T100027_con/RT_T33TVM_A010844_20170720T100027_stack_NDVI_sub.tif"

# %%
# Open images
try:
    file_1_src = gdal.Open(file_1)
except:
    print('Could not open:', file_1)
    sys.exit(1)
try:
    file_1_ndvi_src = gdal.Open(file_1_ndvi)
except:
    print('Could not open:', file_1_ndvi)
    sys.exit(1)
try:
    file_1_udm_src = gdal.Open(file_1_udm)
except:
    print('Could not open:', file_1_udm)
    sys.exit(1)
try:
    file_2_src = gdal.Open(file_2)
except:
    print('Could not open:', file_2)
    sys.exit(1)
try:
    file_2_ndvi_src = gdal.Open(file_2_ndvi)
except:
    print('Could not open:', file_2_ndvi)
    sys.exit(1)

# %%
# Find overlap
im_bb, im_res, im_srs = sits.image_raster_overlap(file_1_src, file_2_src)

# %%
# Resample images
im_1_ndvi = sits.image_raster_resample(file_1_ndvi_src, im_bb, im_res, im_srs)
im_2_ndvi = sits.image_raster_resample(file_2_ndvi_src, im_bb, im_res, im_srs)

# %%
# Show image
sits.image_show_band(im_1_ndvi)
sits.image_show_band(im_2_ndvi)

# %%
# Resample original images
im_1 = sits.image_raster_resample(file_1_src, im_bb, im_res, im_srs)
# im_1_udm = image_raster_resample(file_1_udm_src, im_bb, im_res, im_srs)
im_2 = sits.image_raster_resample(file_2_src, im_bb, im_res, im_srs)

# %%
sits.image_show(im_1)
sits.image_show(im_2)
sits.image_show_band(im_1, band=1, cmap=plt.cm.RdYlGn)

# %%
# # Mask images
# im_1 = planetscope_mask(im_1, im_1_udm)
# image_show(im_1)

# %%
# Create scatterplot
sits.image_scatterplot(im_1, im_2[[1, 2, 3, 8],:,:], im1_in_name="Planet", im2_in_name="Sentinel")
sits.image_scatterplot(im_1_ndvi, im_2_ndvi, im1_in_name="Planet", im2_in_name="Sentinel")

# %%
# Find Pearson correlation coefficient

# %%
# Find Spearman's rank correlation coefficient
