import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from musicalrobot import DES

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
def des_generator(min_comps, max_comps, samples, trials):
    '''Generates list of mol fractions with any amount of samples'''
    #The minimum mole fractions of each component.
    lower_bounds = np.array(min_comps)
    upper_bounds = np.array(max_comps)

    #Generating random DES compositions within the design contraints, mole fractions of each composition must = 1.
    DES_trials = np.random.rand(trials*20,2)
    DES_trials = DES_trials*(upper_bounds-lower_bounds)+lower_bounds

    #Adding mole fractions of each component in the random trial.
    mole_sum = np.sum(DES_trials, axis=1)
    DES_samples = DES_trials/mole_sum[:,None]

    #This normalization may still lead to compositions that do not satisfy the constraint
    upper_check = DES_samples>upper_bounds
    lower_check = DES_samples<lower_bounds
    combined_check = np.append(upper_check, lower_check, axis=1)
    SafeList = np.any(combined_check, axis=1)
    DeleteList = ~SafeList
    Feasible_DES_samples = DES_samples[DeleteList,:]
    print(" "+str(len(Feasible_DES_samples))+" feasible DES samples generated, clustered into "+str(samples)+" samples")

    #Apply K-means clustering to DES samples
    kmeans = KMeans(n_clusters=samples, random_state=0).fit(Feasible_DES_samples)
    DES_molfrac = kmeans.cluster_centers_
    return DES_molfrac


# Function: Converts mol fractions to volumes for pipetting and reference
# Step: create system of equations, solve system to determine the volume of each
# Input: DES mol fractions, desired vol, stock solutions
# Output: list of volumes of each component for the desired volume
def mol_to_vol(DES_molfrac, stock, volume):
    '''Converts mol fractions to volumes depending on desired volume and stocks'''
    # pre-processing
    number = len(stock)
    samples = len(DES_molfrac)
    size = (samples, number)
    final_vol = np.zeros(size)


    count = 0
    for row in DES_molfrac:
        def f(x):
            for i in range(number):
                total = []
                place = stock[i]*x[i]
                total.append(place)
            total = sum(total)
            y = np.zeros(np.size(x))
            y[0] = x[0] + x[1] - volume
            for i in range(number-1):
                y[i+1] = (((stock[i])*x[i])/(total)) - row[i]
            return y
        x0 = np.array([100.0, 100.0, 100.0])
        x = fsolve(f, x0)
        for i in range(number):
            final_vol[count, i] = x[i]
        count = count + 1
    return final_vol


# Function: Convert list of volumes into a list that opentrons can use
# Step: separate lists
# Input: final_vol
# Output: open_vol - usuable by opentrons machine
def open_vol(final_vol):
    '''Converts the array of volumes with comp volumes to two lists of separate volumes'''
    number = len(final_vol[0])
    open_vol = []

    for i in range(number):
        string = "comp" + str(i)
        string = []
        for row in final_vol:
            hold = row[i]
            string.append(hold)
        open_vol.append(string)
    return open_vol
