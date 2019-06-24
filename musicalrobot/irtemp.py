import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    cels = irtemp.centikelvin_to_celsius(temp)
    fahr = cels * 9 / 5 + 32
    return cels, fahr