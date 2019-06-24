import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import DES

from sklearn.cluster import KMeans
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Function: General Des generator - any number of components will work
# Step: Generate a large amount of possibilities, narrow down within constraints
# Input: min and max mol fraction desired, desired output samples, with trial estimate
# Output: list of mol fractions for each component
def test_des_generator():
    '''Test: Generates list of mol fractions with any amount of samples'''
    # inputs
    min_comps = [.2, .2]
    max_comps = [.8, .8]
    samples = 5
    trials = 10
    #running
    DES_molfrac = DES.des_generator(min_comps, max_comps, samples, trials)
    #asserts
    assert isinstance(DES_molfrac, np.ndarray),'Output is not a array'
    assert len(max_comps) == len(DES_molfrac[0]), 'Component space is not equal'
    return

__
# Function: Converts mol fractions to volumes for pipetting and reference
# Step: create system of equations, solve system to determine the volume of each
# Input: DES mol fractions, desired vol, stock solutions
# Output: list of volumes of each component for the desired volume
def test_mol_to_vol():
    '''Test: Converts mol fractions to volumes depending on desired volume and stocks'''
    # inputs
    min_comps = [.2, .2]
    max_comps = [.8, .8]
    samples = 5
    trials = 10

    stock = [2, 4] #molarity
    volume = 150 #ml

    DES_molfrac = DES.des_generator(min_comps, max_comps, samples, trials)
    #running
    final_vol = DES.mol_to_vol(DES_molfrac, stock, volume)
    # asserts
    assert isinstance(final_vol, np.ndarray),'Output is not a array'
    assert len(max_comps) == len(final_vol[0]), 'Component space is not equal'
    assert sum(final_vol[0]) == volume, 'Volume is not as desired'
    return


# Function: Convert list of volumes into a list that opentrons can use
# Step: separate lists
# Input: final_vol
# Output: open_vol - usuable by opentrons machine
def test_open_vol():
    '''Test: Converts the array of volumes with comp volumes to two lists of separate volumes'''
    # inputs
    min_comps = [.2, .2]
    max_comps = [.8, .8]
    samples = 5
    trials = 10

    stock = [2, 4] #molarity
    volume = 150 #ml

    DES_molfrac = DES.des_generator(min_comps, max_comps, samples, trials)
    final_vol = DES.mol_to_vol(DES_molfrac, stock, volume)
    # running
    open_vol = DES.open_vol(final_vol)
    # asserts
    assert isinstance(open_vol, np.ndarray),'Output is not a array'
    return
