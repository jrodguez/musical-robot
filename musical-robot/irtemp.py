
import pandas as pd
import numpy as np
from skimage import io

__
# Function:
# Step:
# Input:
# Output:
def name(input):
    '''Doc String'''
    return output
__


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
def name(frames, factor):
    '''Loops through all frames in the video clip to portion out chosen frames'''
    chosenframes = []
    frame = 0
    #the frame rate for a Letpton FLRS camera is 27 fps.
    while frame <= frames:
        if frame % factor == 0:
            chosenframes.append(frame)
            frame = frame + 1
        else:
            frame = frame + 1
    return chosenframes

# Function:
# Step:
# Input:
# Output:
def name(input):
    '''Doc String'''
    return output
