import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import irtemp
import edge_detection

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