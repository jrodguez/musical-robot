import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irtemp

import pandas as pd
import numpy as np
from skimage import io

# __
# # # Function:
# # Step:
# # Input:
# # Output:
# def name(input):
#     '''Doc String'''
#     return output
# __


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
    cels = irtemp.centikelvin_to_celsius(temp)
    fahr = cels * 9 / 5 + 32
    return cels, fahr


# Function: reads the image data and saves the values to variables
# Step: using skimage read the shape of the tiff file
# Input: raw image file (usually in tiff format)
# Output: integer values of frames, height(rows), and width(columns)
def image_read(image):
    '''Reads the image data and saves the values to variables.'''
    frames, height, width = image.shape
    return frames, height, width

# Function: Loops through all frames in the video clip to portion out chosen frames
# Step: iterate through all frames checking at intervals of the factor, saving
# the idexes of the chosen frames into list
# Input: number of all frames, desired integer to factor by
# Output: list of indexes of chosen frames
def frame_loop(frames, factor):
    '''Loops through all frames in the video clip to portion out chosen frames'''
    chosenframes = []
    frame = 0
    #the frame rate for a Letpton FLRS camera is 27 fps.
    while frame <= frames - 1:
        if frame % factor == 0:
            chosenframes.append(frame)
            frame = frame + 1
        else:
            frame = frame + 1
    return chosenframes

# Function:Loops through alle the frames and calculates the time since starting
# Step: create list of all times, divide frame by frame rate
# Input: list of chosen frames
# Output: list of the corresponding times with the frames
def time_index(chosenframes):
    '''Creates an index of time with the specific chosen chosenframes'''
    #assumption is that there is 27 frames per second
    alltime = []
    maxlength = len(chosenframes)
    index = 0

    while index < maxlength:
        frame = chosenframes[index]
        time = frame / 27
        alltime.append(round(time, 2))
        index = index + 1
    return alltime

# Function: Creates data frame of all of the data
# Step: input all of the output data sets, concatenate
# Input: chosen frames, times, temperatures (celcius and fahrenheit)
# Output: data frame of all data sets
def dataframe_create(chosenframes, alltime, alltempc, alltempf):
    '''Create a data frame of all of the inputted data sets'''
    data = pd.DataFrame()
    data['Frame'] = chosenframes
    data['Time'] = alltime
    data['Temp (C)'] = alltempc
    data['Temp (F)'] = alltempf
    return data

# Function: Reads the temperature for every point in the frame and creates array
# Step:set constants, iterate over columns and rows reading all tempertures
# Input:single frame of image
# Output: temperatures at every point in array
def all_temp_single_frame(image):
    '''Reads all temperatures in a single frame'''
    frames, height, width = irtemp.image_read(image)
    alltempall = np.zeros((height, width), dtype = float)
    row = 0
    col = 0

    while col < height :
        row = 0
        while row < width:
            temp = image[col, row]
            cels = irtemp.centikelvin_to_celsius(temp)
            alltempall[col, row] = cels
            row += 1
        col += 1
    return alltempall

# Function: finds a single temperature at a specifc point over all frames
# Step:set arrays, index the frames wanted, iterate over frames, determine temp at points
# Input: the frames that you want, the col and row desired, and full image
# Output: lists of the temperatures in celcius and fahrenheit
def single_temp_all_frame(chosenframes, col, row, image):
    '''Finds a single temperature over all chosen frames'''
    maxlength = len(chosenframes)
    alltempc = []
    alltempf = []
    index = 0

    while index < maxlength:
        frame = chosenframes[index]
        framearray = image[frame]
        temp = framearray[col,row]
        cels, fahr = to_temperature(temp)
        alltempc.append(cels)
        alltempf.append(fahr)
        index = index + 1
    return alltempc, alltempf
