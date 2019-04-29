import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from skimage import io

import irtemp




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

def test_image_read():
    '''Test: Reads the image data and saves the values to variables.'''
    #inputs
    image = io.imread('../../doc/Lepton_Capture.tiff')
    #running functionS
    frames, height, width = irtemp.image_read(image)
    #asserts
    assert isinstance(frames, int), 'Output is not a integer'
    assert isinstance(height, int), 'Output is not a integer'
    assert isinstance(width, int), 'Output is not a integer'
    assert isinstance(image, np.ndarray), "Input is not a numpy array"
    return

def test_frame_loop():
    '''Test: Loops through all frames in the video clip to portion out chosen frames'''
    #inputs
    image = io.imread('../../doc/Lepton_Capture.tiff')
    frames, height, width = irtemp.image_read(image)
    factor = 27 #defined by number of frames desired
    #running function
    chosenframes = irtemp.frame_loop(frames, factor)
    #asserts
    assert isinstance(chosenframes, list), "Output is not a list"
    return

def test_time_index():
    '''Test: Creates an index of time with the specific chosen chosenframes'''
    #inputs
    image = io.imread('../../doc/Lepton_Capture.tiff')
    frames, height, width = irtemp.image_read(image)
    factor = 27
    chosenframes = irtemp.frame_loop(frames, factor)
    #running function
    alltime = irtemp.time_index(chosenframes)
    #asserts
    assert isinstance(alltime, list), "Output is not a list"
    return

def test_dataframe_create():
    '''Test: Create a data frame of all of the inputted data sets'''
    #inputs
    factor = 27
    col = 5
    row = 5

    image = io.imread('../../doc/Lepton_Capture.tiff')
    frames, height, width = irtemp.image_read(image)
    chosenframes = irtemp.frame_loop(frames, factor)
    alltime = irtemp.time_index(chosenframes)
    alltempc, alltempf = irtemp.single_temp_all_frame(chosenframes, col, row, image)
    #running function
    data = irtemp.dataframe_create(chosenframes, alltime, alltempc, alltempf)
    #asserts
    assert isinstance(alltime, list), "Input is not a list"
    assert isinstance(alltempc, list), "Input is not a list"
    assert isinstance(alltempf, list), "Input is not a list"
    assert isinstance(chosenframes, list), "Input is not a list"
    assert isinstance(data, pd.core.frame.DataFrame), "Output is not a dataframe"
    return


def test_all_temp_single_frame():
    '''Test: Reads all temperatures in a single frame'''
    #inputs
    image = io.imread('../../doc/Lepton_Capture.tiff')
    #running function
    alltempall = irtemp.all_temp_single_frame(image)
    #asserts
    assert isinstance(alltempall, np.ndarray), "Output is not an array"
    return

def test_single_temp_all_frame():
    '''Test: Finds a single temperature over all chosen frames'''
    #inputs
    factor = 27
    col = 5
    row = 5

    image = io.imread('../../doc/Lepton_Capture.tiff')
    frames, height, width = irtemp.image_read(image)
    chosenframes = irtemp.frame_loop(frames, factor)
    #running function
    alltempc, alltempf = irtemp.single_temp_all_frame(chosenframes, col, row, image)
    #asserts
    assert isinstance(alltempc, list), "Output is not a list"
    assert isinstance(alltempf, list), "Output is not a list"
    return
