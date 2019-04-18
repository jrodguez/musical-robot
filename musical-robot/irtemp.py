
import pandas as pd
import numpy as np

def centikelvin_to_celsius(temp):
    cels = (temp - 27315)/100
    return cels