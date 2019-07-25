import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import numpy as np


# Function 1: Converts the raw centikelvin reading to Celcius
# Step: convert using given formula for centikelvin to celcius
# Input: centikelvin reading
# Output: float value in celcius
def centikelvin_to_celsius(temp):
    '''Converts given centikelvin value to Celsius'''
    cels = (temp - 27315)/100
    return cels

# Function: Converts raw centikelvin reading to fahrenheit
# Step:Use function (1) to convert to cels, use equation to convert to fahr
# Input: centikelvin reading
# Output: float value in fahrenheit
def to_fahrenheit(temp):
    '''Converts given centikelvin reading to fahrenheit'''
    cels = centikelvin_to_celsius(temp)
    fahr = cels * 9 / 5 + 32
    return fahr

# Function: Covnerts raw centikelvin value to both fahrenheit and celcius
# Step: Use function (1) to convert to cels, use equation to convert to fahr
# Input: centikelvin reading
# Output: float values in celcius and fahrenheit
def to_temperature(temp):
    '''Converts given centikelvin value to both fahrenheit and celcius'''
    cels = centikelvin_to_celsius(temp)
    fahr = cels * 9 / 5 + 32
    return cels, fahr


# Function: Determines the slope between temperature points with some smoothing
# Step: append all slopes and index points to variables, using given jump distance
# Input: jump, the sample temp data set for given sample
# Output: slope for each requested jump, index for each requested jump
def slope_gen(sample_temp, plate_temp, jump):
    '''Determines the slope between temperature points with some smoothing'''
    # reccomended jump higher than 15
    y = sample_temp
    x = plate_temp
    all_slope = []
    all_index = []
    index = 0

    while index < len(y):
        if index >= len(y) - jump:
            break

        if x[index]-x[index+jump] == 0:
            jump = jump + 1
            index = 0
        else:
            pass

        slope = (y[index]-y[index + jump])/(x[index]-x[index + jump])

        all_slope.append(slope)
        all_index.append(index)
        index = index + jump

    return all_slope, all_index

# Function: Smoothes the results generated in slope gen to coax the max dense zones out
# Step: average the slope and index over a range of values to produce a more averaged graph
# Input: reults of slope_gen (all_slopes and all indexes in the set), smoothing factor
# Output: the resulting smoothed index and the slopes
def smoothing_slope(all_slope, all_index, factor):
    '''Smoothes the results generated in slope gen to coax the max dense zones out'''
    #smoothing factor allows for more or less smoothing, reccomend around 5
    smooth_slope = []
    smooth_index = []

    for i in range(len(all_slope)):
        smooth_slope.append(sum(all_slope[i:i+factor])/factor)
        smooth_index.append(sum(all_index[i:i+factor])/factor)
    return smooth_slope, smooth_index

# Function: Produces temperature value of the most possible inflection point
# Step: set constraints on the ideal slope, generate list of possible, average
# Input: smooth_slope and smooth_index
# Output: temperature of most likely, list of other close/possible points
def inflection_temp(smooth_slope, smooth_index, sample_temp):
    '''Produces temperature value of the most possible inflection point'''
    yposs =[]
    index_poss = []

    for i in range(len(smooth_slope)):
        upbound = 2*np.std(smooth_slope) + np.mean(smooth_slope)
        downbound = np.mean(smooth_slope) - 2 * np.std(smooth_slope)
    if smooth_slope[i] < downbound:
        yposs.append(smooth_slope[i])
        index_poss.append(smooth_index[i])
    elif smooth_slope[i] > upbound:
        yposs.append(smooth_slope[i])
        index_poss.append(smooth_index[i])
    else:
        pass
    if len(index_poss) == 0:
        print('no inflection point found')
        melt_temp = False
        return melt_temp, index_poss
    else:
        pass
    check = int(np.mean(index_poss))
    melt_temp = sample_temp[check]
    return melt_temp, index_poss


# Function:Wrapping function for single sample
# Step: combines all of the following inflection point functions into one
# Input: sample temperatures, desired intial jump, smoothing factor
# Output: melting temperature and other possible values if determined to be incorrect
def melting_temperature(sample_temp, plate_temp, jump, factor):
    '''Wraps all of the inflection point functions to a single one'''
    all_slope, all_index = slope_gen(sample_temp, plate_temp, jump)
    smooth_slope, smooth_index = smoothing_slope(all_slope, all_index, factor)
    melt_temp, index_poss = inflection_temp(smooth_slope, smooth_index, sample_temp)
    return melt_temp, index_poss


# Function: Looping through all samples to determine all melting temperatures
# Step: loop through the different samples running the wrapped melting temp function for each one
# Input: all sample temperatures for all samples
# Output: all sample melting temperature, all possible indexes of other melting temps
def all_melting(all_sample_temp, plate_temp, jump, factor):
    '''Looping through all samples to determine all melting temperatures'''
    all_melt = []
    all_possible = []

    for i in range(len(all_sample_temp)):
        sample_temp = all_sample_temp[i]
        hold_melt, hold_possible = melting_temperature(sample_temp, plate_temp, jump, factor)

        all_melt.append(hold_melt)
        all_possible.append(hold_possible)
    return all_melt, all_possible
