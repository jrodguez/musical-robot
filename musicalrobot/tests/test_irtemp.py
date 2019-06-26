
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import irtemp

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


# Function: Determines the slope between temperature points with some smoothing
# Step: append all slopes and index points to variables, using given jump distance
# Input: jump, the sample temp data set for given sample
# Output: slope for each requested jump, index for each requested jump
def test_slope_gen():
    '''Test:Determines the slope between temperature points with some smoothing'''
#inputs
    sample_temp = [3,5,6,7,2,7,2,4]
    plate_temp = [5,2,5,7,6,7,3,5]
    jump = 2
#running function
    all_slope, all_index = irtemp.slope_gen(sample_temp, plate_temp, jump)
#asserts
    return

# Function: Smoothes the results generated in slope gen to coax the max dense zones out
# Step: average the slope and index over a range of values to produce a more averaged graph
# Input: reults of slope_gen (all_slopes and all indexes in the set), smoothing factor
# Output: the resulting smoothed index and the slopes
def test_smoothing_slope():
    '''Smoothes the results generated in slope gen to coax the max dense zones out'''
    #inputs
    sample_temp = [3,5,6,7,2,7,2,4]
    plate_temp = [5,2,5,7,2,7,3,5]
    jump = 2
    factor = 3
    all_slope, all_index = irtemp.slope_gen(sample_temp, plate_temp, jump)
    #running function
    smooth_slope, smooth_index = irtemp.smoothing_slope(all_slope, all_index, factor)
    #asserts
    return

# Function: Produces temperature value of the most possible inflection point
# Step: set constraints on the ideal slope, generate list of possible, average
# Input: smooth_slope and smooth_index
# Output: temperature of most likely, list of other close/possible points
def test_inflection_temp():
    '''Produces temperature value of the most possible inflection point'''
    #inputs
    sample_temp = [3,5,6,7,2,7,2,4]
    plate_temp = [5,2,5,7,2,7,3,5]
    jump = 2
    factor = 3
    all_slope, all_index = irtemp.slope_gen(sample_temp,plate_temp, jump)
    smooth_slope, smooth_index = irtemp.smoothing_slope(all_slope, all_index, factor)
    #running function
    melt_temp, index_poss = irtemp.inflection_temp(smooth_slope, smooth_index, sample_temp)
    #asserts
    return


# Function:Wrapping function for single sample
# Step: combines all of the following inflection point functions into one
# Input: sample temperatures, desired intial jump, smoothing factor
# Output: melting temperature and other possible values if determined to be incorrect
def test_melting_temperature():
    '''Wraps all of the inflection point functions to a single one'''
    #inputs
    sample_temp = [3,5,6,7,2,7,2,4]
    plate_temp = [5,2,5,7,2,7,3,5]
    jump = 2
    factor = 3
    #running function
    melt_temp, index_poss = irtemp.melting_temperature(sample_temp, plate_temp, jump, factor)
    #asserts
    return


# Function: Looping through all samples to determine all melting temperatures
# Step: loop through the different samples running the wrapped melting temp function for each one
# Input: all sample temperatures for all samples
# Output: all sample melting temperature, all possible indexes of other melting temps
def test_all_melting():
    '''Looping through all samples to determine all melting temperatures'''
    #inputs
    all_sample_temp = [[3,5,6,7,2,7,2,4], [4,1,6,7,2,4,6,1]]
    plate_temp = [5,2,5,7,2,7,3,5]
    jump = 2
    factor = 3
    #running function
    all_melt, all_possible = irtemp.all_melting(all_sample_temp, plate_temp, jump, factor)
    #asserts
    return all_melt, all_possible
