
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
