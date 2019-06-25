import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import irtemp
import edge_detection
import pixel_analysis

from irtemp import centikelvin_to_celsius
from skimage import io
from skimage import feature
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects  
from scipy.signal import find_peaks


# def test_name():
#     '''Doc String'''
#     #inputs
#     #running function
#     #asserts
#     return



##################### Peak detection and pixel analysis function #######################################

def test_image_eq():
    ''' Test for fucntion which equalizes a low contrast image'''
    pixel_frames = edge_detection.input_file('../musical-robot/musicalrobot/data/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = pixel_analysis.image_eq(n_frames,pixel_frames)
    assert isinstance(img_eq,np.ndarray),'Output is not an array'
    assert pixel_frames[0].shape == img_eq.shape, 'Output array shape is not same as the input array shape.'
    return

def test_pixel_sum():
    '''Test for function which obtains the sum of pixels over all rows and columns'''
    pixel_frames = edge_detection.input_file('../musical-robot/musicalrobot/data/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = pixel_analysis.image_eq(n_frames,pixel_frames)
    column_sum, row_sum = pixel_analysis.pixel_sum(img_eq)
    assert isinstance(column_sum,list),'Column sum is not a list'
    assert isinstance(row_sum,list),'Row sum is not a list'
    assert len(row_sum) == img_eq.shape[0], 'The length of row_sum is not equal to number of rows in the input image'
    assert len(column_sum) == img_eq.shape[1], 'The length of column_sum is not equal to number of columns in the input image'
    return

def test_peak_values():
    '''Test for function which finds peaks from the column_sum and row_sum arrays
        and return a dataframe with sample locations and plate locations.'''
    pixel_frames = edge_detection.input_file('../musical-robot/musicalrobot/data/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = pixel_analysis.image_eq(n_frames,pixel_frames)
    column_sum, row_sum = pixel_analysis.pixel_sum(img_eq)
    n_columns = 12
    n_rows = 8
    sample_location = pixel_analysis.peak_values(column_sum,row_sum,n_columns,n_rows,img_eq)
    assert isinstance(sample_location,pd.DataFrame),'Output is not a dataframe'
    assert len(sample_location)==n_columns*n_rows, 'Wrong number of sample locations are present'
    return

def test_pixel_intensity():
    '''Test for function which determines sample temperature and plate temperature'''
    pixel_frames = edge_detection.input_file('../musical-robot/musicalrobot/data/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = pixel_analysis.image_eq(n_frames,pixel_frames)
    column_sum, row_sum = pixel_analysis.pixel_sum(img_eq)
    n_columns = 12
    n_rows = 8
    sample_location = pixel_analysis.peak_values(column_sum,row_sum,n_columns,n_rows,img_eq)
    x_name = 'X'
    y_name = 'Y'
    plate_name = 'plate_location'
    pixel_sample,pixel_plate = pixel_analysis.pixel_intensity(sample_location, pixel_frames, x_name,y_name,plate_name)
    assert isinstance(pixel_sample,list),'Output is not a list'
    assert isinstance(pixel_plate,list),'Output is not a list'
    assert len(pixel_sample)==n_columns*n_rows,'Temperature obtained for wrong number of samples'
    assert len(pixel_plate)==n_columns*n_rows,'Temperature obtained for wrong number of plate locations'
    return

def test_pixel_temp():
    '''Test for the wrapping function'''
    pixel_frames = edge_detection.input_file('../musical-robot/musicalrobot/data/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    n_columns = 12
    n_rows = 8
    temp, plate_temp = pixel_analysis.pixel_temp(pixel_frames,n_frames,n_columns,n_rows)
    assert isinstance(temp,list),'Output is not a list'
    assert isinstance(plate_temp,list),'Output is not a list'
    assert len(temp)==n_columns*n_rows,'Temperature obtained for wrong number of samples'
    assert len(plate_temp)==n_columns*n_rows,'Temperature obtained for wrong number of plate locations'
    return