# coding: utf-8

# Satellite Image Time Series Analysis
#
# Krištof Oštir
# UL FGG
# (c) 2020

# Imports
import gdal
import osr
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys


# Find overlapping area, determine minimum resolution and resample images
def image_raster_overlap(im1, im2, debug=1):
    """
    image_raster_overlap: Open raster images, find overlapping area, determine
     minimum resolution and resample both images. Function returns the overlapping
     bounding box, minimum resolution and coordinate system.

    Args:
        im1: GDAL opened raster image
        im2: GDAL opened raster image

    Returns:
        stack_bb: common bounding box
        stack_res_ minimum resolution
        stack_srs: EPSG coordinate system
    """

    # Check if both rasters have same SRS
    im1_proj = osr.SpatialReference(wkt=im1.GetProjection()) \
        .GetAttrValue('AUTHORITY', 1)
    # print(im1_proj.GetAttrValue('AUTHORITY', 1))
    im2_proj = osr.SpatialReference(wkt=im2.GetProjection()) \
        .GetAttrValue('AUTHORITY', 1)
    # print(im2_proj.GetAttrValue('AUTHORITY', 1))
    if im1_proj != im2_proj:
        print("Files have different projections")
        print("EPSG:", im1_proj, im1)
        print("EPSG:", im2_proj, im2)
        return
    else:
        stack_srs = 'EPSG:' + im1_proj
        print(stack_srs)

    # Find overlap bounding box and resolution
    im1_gt = im1.GetGeoTransform()
    im2_gt = im2.GetGeoTransform()
    # Find each image's bounding box
    im1_bb = [im1_gt[0],
              im1_gt[3] + (im1_gt[5] * im1.RasterYSize),
              im1_gt[0] + (im1_gt[1] * im1.RasterXSize),
              im1_gt[3]]
    im2_bb = [im2_gt[0],
              im2_gt[3] + (im2_gt[5] * im2.RasterYSize),
              im2_gt[0] + (im2_gt[1] * im2.RasterXSize),
              im2_gt[3]]
    # Overlap
    stack_bb = [max(im1_bb[0], im2_bb[0]),
                max(im1_bb[1], im2_bb[1]),
                min(im1_bb[2], im2_bb[2]),
                min(im1_bb[3], im2_bb[3])]

    if stack_bb[0] > stack_bb[2] or stack_bb[1] > stack_bb[3]:
        print("Images do not overlap")
        print(im1_gt, im1)
        print(im2_gt, im2)
        return
    if (debug==1):
        print('Overlapping bounding box: %s' % str(stack_bb))

    # Stack image size and resolution
    stack_res = min(im1_gt[1], im2_gt[1])
    stack_x = int((stack_bb[2] - stack_bb[0]) / stack_res)
    stack_y = int((stack_bb[3] - stack_bb[1]) / stack_res)
    if (debug==1):
        print("Stack resolution:", stack_res)
        print("Stack dimensions:", stack_x, stack_y)

    return stack_bb, stack_res, stack_srs


# Read image
def image_raster_read(im):
    """
    image_raster_read: Read image to numpy array

    Args:
        im: GDAL opened raster image

    Returns:
        im_data: image numpy array
    """

    # Find nodata value
    srcband = im.GetRasterBand(1)
    nodata = srcband.GetNoDataValue()

    # Read array
    im_data = im.ReadAsArray()
    # Find nodata value
    im_mask = im_data == nodata
    im_data = ma.masked_array(im_data, mask=im_mask)

    return im_data


# Resample image
def image_raster_resample(im, im_bbox, im_res, im_srs, res_method="near"):
    """
    image_raster_resample: Crop image to bounding box and resample it. Function returns numpy
     array of the cropped image data. More information is available at:
     https://www.gdal.org/gdalwarp.html

    Args:
        im: GDAL opened raster image
        im_bbox: bounding box
        im_res: resolution
        im_srs: EPSG coordinate system
        res_method: resampling method, default is nearest neighbor

    Returns:
        im_res_data: resampled image
    """

    # Read image, resample
    im_res = gdal.Warp('', im,
                       dstSRS=im_srs, format='VRT', outputBounds=im_bbox,
                       xRes=im_res, yRes=im_res,
                       resampleAlg=res_method)
    # Find nodata value
    srcband = im.GetRasterBand(1)
    nodata = srcband.GetNoDataValue()

    # Read array
    im_res_data = im_res.ReadAsArray()
    # Find nodata value
    im_mask = im_res_data == nodata
    im_res_data = ma.masked_array(im_res_data, mask=im_mask)

    return im_res_data 

# TODO create function for masking
def mask_data(im, im_udm, value=0):
    """
    mask_data: Apply value mask to an array. All areas with different value are masked with NaN. Example: clouds and unusable data.
    More information About the mask is available in:
    https://www.planet.com/products/satellite-imagery/files/1610.06_Spec%20Sheet_Combined_Imagery_Product_Letter_ENGv1.pdf
    Or apply STORM mask to Sentinel-2 image. All areas with clouds and unusable data are
    masked with NaN. More information About the mask is available in txt files with the processed images.

    Args:
        im: image as numpy array, 16-bit
        im_udm: mask as numpy array 8-bit

    Returns:
        im_masked: masked image
    """
    
    # Mask image
    im_out = np.where(im_udm == value, im, np.nan)
    
    return im_out



# TODO create function for scatterplot
def image_scatterplot(im1_in, im2_in, file="", sample=10000, im1_in_name="", im2_in_name="", **kwargs):
    """
    image_scatterplot: Display scatterplot of bands beetween images.

    Args:
        im1: first image as numpy array
        im2: second image as numpy array
        file: name of PDF to store scatterplot
        sample: number of sample points
    """

    # Prepare data
    # Check if images are of same size
    if im1_in.shape != im2_in.shape:
        print("Images are not of same size.")
        print(im1_in.shape, im2_in.shape)
        sys.exit(1)
    elif len(im1_in.shape) == 3:
        im_bands = im1_in.shape[0]
        im_size = im1_in.shape[1:2]
    else:
        im_bands = 1
        im_size = im1_in.shape
        im1_in = ma.array([im1_in])
        im2_in = ma.array([im2_in])
    # Check if data is integer, convert to float
    if im1_in.dtype != float:
        im1_in = im1_in.astype(float)
    if im2_in.dtype != float:
        im2_in = im2_in.astype(float)
    # Mask nodata
    # TODO they are already masked, filled is not supported in numpy
    #im1_in = im1_in.filled(np.nan)
    #im2_in = im2_in.filled(np.nan)

    # TODO check if output is to file

    # Plot parameters
    # TODO Check cmap
    cmap = "gray"

    # Determine and set plot limits, by image
    im1_lim = [np.nanquantile(im1_in, 0.01), np.nanquantile(im1_in, 0.99)]
    im2_lim = [np.nanquantile(im2_in, 0.01), np.nanquantile(im2_in, 0.99)]

    for band in range(im_bands):
        im1 = im1_in[band, :, :]
        im2 = im2_in[band, :, :]

        # Band name
        band_name = " B" + str(band + 1)

        # Create plot
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)

        # Plot first image
        axis1 = fig.add_subplot(311)
        axis1.title.set_text(im1_in_name)
        axis1.imshow(im1, cmap=cmap, clim=im1_lim)

        # Plot second image
        axis2 = fig.add_subplot(312)
        axis2.title.set_text(im2_in_name)
        axis2.imshow(im2, cmap=cmap, clim=im2_lim)

        # TODO Flatten the array, remove no-data
        im1_flat = im1.flat
        im2_flat = im2.flat
        # Check mask
        idx = ~np.isnan(im1_flat)
        im1_flat = im1_flat[idx]
        im2_flat = im2_flat[idx]
        idx = ~np.isnan(im2_flat)
        im1_flat = im1_flat[idx]
        im2_flat = im2_flat[idx]

        # Aggregate, by factor
        idx = np.random.choice(np.arange(len(im1_flat)), sample, replace=False)
        im1_sample = im1_flat[idx]
        im2_sample = im2_flat[idx]

        # Linear regression
        m, b = np.polyfit(im1_sample, im2_sample, 1)

        # Create scatter plot
        axis3 = fig.add_subplot(313, aspect='equal')
        axis3.title.set_text('Scatterplot')
        axis3.set(xlabel=im1_in_name + band_name, ylabel=im2_in_name + band_name)
        axis3.set_xlim(im1_lim)
        axis3.set_ylim(im2_lim)
        axis3.scatter(im1_sample, im2_sample, marker=".", s=1)
        axis3.plot(im1_sample, m * im1_sample + b, '-', c="red")

        # Show image
        fig.show()
        if file!="":
            fig.savefig(file+"_"+band_name)
        
        # # Correlation coefficient
        # np.corrcoef(ps_ndvi_sample, s2_ndvi_sample)

    return


def image_show(im, bands=[2, 1, 0], scale="auto", **kwargs):
    """
    image_show: Show RGB image stored in numpy array

    Args:
        im_in: image as numpy array
        bands: bands to use as blue, green, red, default is 2, 1, 0
        scale: scale no, auto, [min, max], default auto
         - image by clipping 1%% of histogram
        **kwargs: arguments passed to plt.image()
    """

    # Get bands
    im_in = im[bands, :, :].astype(float)

    # Set scale factor for data type per band
    if type(scale) == list and len(scale) == 2:
        scale_max = scale[1]
        scale_min = scale[0]
        scale_f = 1 / scale_max
        im_in = (im_in - scale_min) * scale_f
    elif type(scale) == str:
        if scale == "auto":
            for band in range(3):
                scale_max = np.quantile(im_in[band], 0.99)
                scale_min = np.quantile(im_in[band], 0.01)
                scale_f = 1 / (scale_max - scale_min)
                im_in[band] = (im_in[band] - scale_min) * scale_f
        elif scale == "no":
            scale_max = im_in.max()
            scale_min = im_in.min()
            scale_f = 1 / scale_max
            im_in = (im_in - scale_min) * scale_f
    else:
        print("Image can not be displayed.")
        print("Wrong parameters.")
        sys.exit(1)

    # Create band combination
    image = np.dstack(im_in)
    image[image > 1] = 1
    image[image < 0] = 0

    # Plot
    plt.imshow(image, **kwargs)
    plt.show()


def image_show_band(im, band=0, scale="auto", **kwargs):
    """
    image_show: Show single band from image stored in numpy array

    Args:
        im: image as numpy array
        band: band to use, default is 0
        scale: scale image to min and max, default is true
        **kwargs: arguments passed to plt.image()
    """

    # Check image size
    im_dims = im.shape
    if len(im_dims) == 2:
        b_grey = im.astype(float)
    elif len(im_dims) == 3:
        b_grey = im[band].astype(float)
    else:
        print('Image has wrong number of bands')
        sys.exit(1)

    # Set scale factor for data type per band
    if type(scale) == list and len(scale) == 2:
        scale_max = scale[1]
        scale_min = scale[0]
        scale_f = 1 / scale_max
        b_grey = (b_grey - scale_min) * scale_f
    elif type(scale) == str:
        if scale == "auto":
            scale_max = np.quantile(b_grey.compressed(), 0.99)
            scale_min = np.quantile(b_grey.compressed(), 0.01)
            scale_f = 1 / (scale_max - scale_min)
            b_grey = (b_grey - scale_min) * scale_f
        elif scale == "no":
            scale_max = b_grey.max()
            scale_min = b_grey.min()
            scale_f = 1 / scale_max
            b_grey = (b_grey - scale_min) * scale_f
    else:
        print("Image can not be displayed.")
        print("Wrong parameters.")
        sys.exit(1)

    # Plot
    plt.imshow(b_grey, **kwargs)
    plt.show()


def image_correlation(im1, im2):
    """
    image_correlation Compute Pearson product-moment correlation 
    coefficients of two single band images

    Args:
        im1: image 1 one band only, 0 is nodata
        im2: image 2 one band only, 0 is nodata

    Returns:
        array: Pearson correlation matrix
    """

    # Prepare data
    # Check if images are of same size
    if im1.shape != im2.shape:
        print("Images are not of same size.")
        print(im1.shape, im2.shape)
        sys.exit(1)
    elif len(im1.shape) == 3:
        print("Images must be single band.")
        print(im1.shape, im2.shape)
        sys.exit(1)
    else:
        im1 = ma.array([im1])
        im2 = ma.array([im2])

    # Check if data is integer, convert to float
    if im1.dtype != float:
        im1 = im1.astype(float)
    if im2.dtype != float:
        im2 = im2.astype(float)

    im1[im1==0] = np.nan
    im2[im2==0] = np.nan

    corr = np.corrcoef(im1.flatten(), im2.flatten())

    return corr


def image_linear_regression(im1, im2):
    """
    image_linear_regression Find linear regression parameters of two single band images

    Args:
        im1: image 1 one band only, 0 is nodata
        im2: image 2 one band only, 0 is nodata

    Returns:
        m, b: Linear regression parameters
    """

    # Prepare data
    # Check if images are of same size
    if im1.shape != im2.shape:
        print("Images are not of same size.")
        print(im1.shape, im2.shape)
        sys.exit(1)
    elif len(im1.shape) == 3:
        print("Images must be single band.")
        print(im1.shape, im2.shape)
        sys.exit(1)
    else:
        im1 = ma.array([im1])
        im2 = ma.array([im2])

    # Check if data is integer, convert to float
    if im1.dtype != float:
        im1 = im1.astype(float)
    if im2.dtype != float:
        im2 = im2.astype(float)
    
    im1[im1==0] = np.nan
    im2[im2==0] = np.nan

    [[m, b], cov] = np.polyfit(im1.flatten(), im2.flatten(), 1, cov=True)

    return m, b, cov

# TODO Stack images
