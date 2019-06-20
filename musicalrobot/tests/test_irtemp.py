import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import irtemp
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


def test_centikelvin_to_celcius():
    '''Test: Converts given centikelvin value to Celsius'''
    cels = irtemp.centikelvin_to_celsius(100000)
    assert isinstance(cels, float),'Output is not a float'
    return

def test_to_fahrenheit():
    '''Test: Converts given centikelvin reading to fahrenheit'''
    fahr = irtemp.to_fahrenheit(100000)
    assert isinstance(fahr, float), 'Output is not a float'
    return

def test_to_temperature():
    '''Test: Converts given centikelvin value to both fahrenheit and celcius'''
    cels, fahr = irtemp.to_temperature(100000)
    assert isinstance(fahr, float), 'Output is not a float'
    assert isinstance(cels, float),'Output is not a float'
    return


##########################################################################################################################################################################
##########################################################################################################################################################################
                                                #######Test functions for Image Processing #########
##########################################################################################################################################################################
##########################################################################################################################################################################


def test_input_file():
    '''Test for function which loads the input file'''
    file_name = ('../musical-robot/doc/PPA_Melting_6_14_19.tiff')
    frames = irtemp.input_file(file_name)
    assert isinstance(frames, np.ndarray),'Output is not an array'
    return

def test_flip_frame():
    '''Test for function which flips the frames horizontally
       and vertically to correct for the mirroring during recording.'''
    file_name = ('../musical-robot/doc/PPA_Melting_6_14_19.tiff')
    frames = irtemp.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[40:100])
    flip_frames = irtemp.flip_frame(crop_frame)
    assert isinstance(flip_frames,list),'Output is not a list'
    return

def test_edge_detection():
    ''' Test for function which detects edges,fills and labels the samples'''
    file_name = ('../musical-robot/doc/PPA_Melting_6_14_19.tiff')
    frames = irtemp.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[40:100])
    flip_frames = irtemp.flip_frame(crop_frame)
    labeled_samples = irtemp.edge_detection(flip_frames)
    assert isinstance(labeled_samples, np.ndarray),'Output is not an array'
    assert flip_frames[0].shape == labeled_samples.shape,'Input and Output array shapes are different.'
    return

def test_regprop():
    '''Test for function which determines centroids of all the samples
    and locations on the plate to obtain temperature from'''
    file_name = ('../musical-robot/doc/PPA_Melting_6_14_19.tiff')
    frames = irtemp.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[40:100])
    flip_frames = irtemp.flip_frame(crop_frame)
    labeled_samples = irtemp.edge_detection(flip_frames)
    n_samples = 5
    regprops = irtemp.regprop(labeled_samples,flip_frames,n_samples)
    assert isinstance(regprops,dict),'Output is not a dictionary'
    assert len(regprops)==len(flip_frames),'The number of dataframes in the dictionary is not equal to number of frames input.'
    for i in range(len(flip_frames)):
        assert len(regprops[i])==n_samples,'Wrong number of samples detected'
    return

def test_sample_temp():
    '''Test for function which obtaines temperature of samples and plate temperature'''
    file_name = ('../musical-robot/doc/PPA_Melting_6_14_19.tiff')
    frames = irtemp.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[40:100])
    flip_frames = irtemp.flip_frame(crop_frame)
    labeled_samples = irtemp.edge_detection(flip_frames)
    n_samples = 5
    regprops = irtemp.regprop(labeled_samples,flip_frames,n_samples)
    temp, plate_temp = irtemp.sample_temp(regprops,flip_frames)
    assert isinstance(temp,list),'Sample temperature output is not a list'
    assert isinstance(plate_temp,list),'Plate temperature output is not a list'
    assert len(temp) == n_samples,'Temperature obtained for wrong number of samples detected'
    assert len(plate_temp) == n_samples,'Temperature obtained for wrong number of plate locations'
    return

##################### Peak detection and pixel analysis function #######################################

def test_image_eq():
    ''' Test for fucntion which equalizes a low contrast image'''
    pixel_frames = irtemp.input_file('../musical-robot/doc/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = irtemp.image_eq(n_frames,pixel_frames)
    assert isinstance(img_eq,np.ndarray),'Output is not an array'
    assert pixel_frames[0].shape == img_eq.shape, 'Output array shape is not same as the input array shape.'
    return

def test_pixel_sum():
    '''Test for function which obtains the sum of pixels over all rows and columns'''
    pixel_frames = irtemp.input_file('../musical-robot/doc/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = irtemp.image_eq(n_frames,pixel_frames)
    column_sum, row_sum = irtemp.pixel_sum(img_eq)
    assert isinstance(column_sum,list),'Column sum is not a list'
    assert isinstance(row_sum,list),'Row sum is not a list'
    assert len(row_sum) == img_eq.shape[0], 'The length of row_sum is not equal to number of rows in the input image'
    assert len(column_sum) == img_eq.shape[1], 'The length of column_sum is not equal to number of columns in the input image'
    return

def test_peak_values():
    '''Test for function which finds peaks from the column_sum and row_sum arrays
        and return a dataframe with sample locations and plate locations.'''
    pixel_frames = irtemp.input_file('../musical-robot/doc/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = irtemp.image_eq(n_frames,pixel_frames)
    column_sum, row_sum = irtemp.pixel_sum(img_eq)
    n_columns = 12
    n_rows = 8
    sample_location = irtemp.peak_values(column_sum,row_sum,n_columns,n_rows,img_eq)
    assert isinstance(sample_location,pd.DataFrame),'Output is not a dataframe'
    assert len(sample_location)==n_columns*n_rows, 'Wrong number of sample locations are present'
    return

def test_pixel_intensity():
    '''Test for function which determines sample temperature and plate temperature'''
    pixel_frames = irtemp.input_file('../musical-robot/doc/CHCl_CA_DES_5_31_19.tiff')
    n_frames = len(pixel_frames)
    img_eq = irtemp.image_eq(n_frames,pixel_frames)
    column_sum, row_sum = irtemp.pixel_sum(img_eq)
    n_columns = 12
    n_rows = 8
    sample_location = irtemp.peak_values(column_sum,row_sum,n_columns,n_rows,img_eq)
    x_name = 'X'
    y_name = 'Y'
    plate_name = 'plate_location'
    pixel_sample,pixel_plate = irtemp.pixel_intensity(sample_location, pixel_frames, x_name,y_name,plate_name)
    assert isinstance(pixel_sample,list),'Output is not a list'
    assert isinstance(pixel_plate,list),'Output is not a list'
    assert len(pixel_sample)==n_columns*n_rows,'Temperature obtained for wrong number of samples'
    assert len(pixel_plate)==n_columns*n_rows,'Temperature obtained for wrong number of plate locations'