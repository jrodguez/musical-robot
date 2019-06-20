import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irtemp

import skimage
import numpy as np
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import feature
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects  
from scipy.signal import find_peaks

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
##########################################################################################################################################################################
##########################################################################################################################################################################
                                                #######Image Processing Functions#########
##########################################################################################################################################################################
##########################################################################################################################################################################
# Function to load the input file
def input_file(file_name):
    ''' To load the imput file as an array.
    Args:
        file_name (string) : Name of the file to be loaded as it is 
        saved on the disk. Provide file path if it is not in
        the same directory as the jupyter notebook.
    Returns:
        In case of a video, returns an array for each frame
        in the video.
        In case of an image, return an array.    
    '''
    frames = io.imread(file_name)
    return frames

# Function to flip the frames horizontally and vertically to correct for the mirroring 
# during recording.
def flip_frame(frames):
    ''' To flip all the loaded frames horizontally and vertically
        to correct for the mirroring during recording.
        Args:
        frames(Array): An array containing an array for each frame
        in the video or just a single array in case of an image.
        Returns:
        Flipped frames that can be processed to get temperature data.
    '''
    flip_frames = []
    for frame in frames:
        f_frame = np.fliplr(frame)
        flip_frames.append(np.flipud(f_frame))
    return flip_frames

# Function to detect edges, fill and label the samples.
def edge_detection(frames):
    ''' To detect the edges of the wells, fill and label them to
    determine their centroids.
    Args:
    frames: The frames to be processed and determine the
    sample temperature from.
    Returns:
    labeled_samples: All the samples in the frame are labeled
    so that they can be used as props to get pixel data from.
    '''
    edges = feature.canny(frames[0]/1500)
    filled_samples = binary_fill_holes(edges)
    cl_samples = remove_small_objects(filled_samples,min_size = 20)
    labeled_samples = label(cl_samples)
    return labeled_samples

# Function to determine centroids of all the samples
def regprop(labeled_samples,frames,n_samples):
    ''' Determines the area and centroid of all samples.
        Args:
        labeled_samples(array): An array with labeled samples.
        flip_frames (array) : Original intensity image to determine
        the intensity at sample centroids.
        n_samples: Number of samples in the video being analyzed.
        Returns:
        A dictionary of dataframe with information about samples in every
        frame of the video.
    '''
    regprops = {} 
    for i in range(len(frames)):
        props = regionprops(labeled_samples, intensity_image=frames[i])
        x = np.zeros(len(props)).astype(int)
        y = np.zeros(len(props)).astype(int)
        area = np.zeros(len(props))
        perim = np.zeros(len(props))
        intensity = np.zeros(len(props),dtype=np.float64)
        plate = np.zeros(len(props),dtype=np.float64)
        plate_coord = np.zeros(len(props))
       
        c = 0
        for prop in props:
            x[c] = int(prop.centroid[0])
            y[c] = int(prop.centroid[1])
            #print(y[c])
            area[c] = prop.area
            perim[c] = prop.perimeter
            intensity[c] = frames[i][x[c]][y[c]]
            plate[c] = frames[i][x[c]][y[c]+10]
            plate_coord[c] = y[c]+10
            c = c + 1
            
        regprops[i] = pd.DataFrame({'X': x, 'Y': y,'Plate':plate,'Plate_coord':plate_coord ,'Area': area,
                                'Perim': perim, 'Mean Intensity': intensity},dtype=np.float64)
        regprops[i].sort_values(['Y','X'],inplace=True)
        if len(regprops[i]) != n_samples:
            print('Wrong number of samples are being detected in frame %d' %i)    
    return regprops

# Function to obtain temperature of samples and plate temp
def sample_temp(regprops,frames):
    ''' Function to concatenate all the obtained temperature data
        from the pixel values into lists.
        Args:
        regprops(dictionary): The dictionary of dataframes containing temperature data.
        frames(array): The array of frames to be processed to obtain temperature data.
        Returns:
        temp(list): Temperature of all the samples in every frame of the video.
        plate_temp(list): Temperature of the plate next to every sample in every
        frame of the video.
    '''
    temp = []
    plate_temp = []
    for j in range(len(regprops[1])):
        temp_well = []
        plate_well_temp = []
        for i in range(len(frames)):
            temp_well.append(centikelvin_to_celsius(regprops[i]['Mean Intensity'][j]))
            plate_well_temp.append(centikelvin_to_celsius(regprops[i]['Plate'][j]))
        temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return temp,plate_temp


##########################################################################################################################################################################
##########################################################################################################################################################################
                                                #######Sample detection using peaks#########
##########################################################################################################################################################################
##########################################################################################################################################################################

# Image equalization
def image_eq(n_frames,frames):
    '''Function to obtained an equalized image using all the frames
       in the video.
       Args:
       n_frames(int): The number of frames in the video.
       frames(List): List of arrays of frames in the video.
       Returns:
       img_eq: Equalized image  
    '''
    for II in range(n_frames):
        frame = frames[II]
        img_eq = (frame - np.amin(frame))/(np.amax(frame)-np.amin(frame))
        if II == 0:
            img_ave = img_eq
        else:
            img_ave = img_ave + img_eq
    img_average = img_ave/n_frames
    img_eq = (img_ave - np.amin(img_ave))/(np.amax(img_ave)-np.amin(img_ave))
    return img_eq

# Function to obtain sum of pixels over all the rows and columns
def pixel_sum(frame):
    ''' Funtion to determine sum of pixels over all the rows and columns
        to obtain plots with peaks at the sample position in the array.
        Args:
        frame(array): An array of an image
        Returns:
        column_sum: Sum of pixels over all the columns
        row_sum: Sum of pixels over all the rows
        Also returns plots of column sum and row sum.
    '''
    rows = frame.shape[0]
    columns = frame.shape[1]
    column_sum = []
    for i in range(0,columns):
        column_sum.append(sum(frame[:,i]))
    row_sum = []
    for j in range(0,rows):
        row_sum.append(sum(frame[j,:]))
    column_sum = [x * -1 for x in column_sum]
    row_sum = [x * -1 for x in row_sum]
    plt.plot(range(len(column_sum)),column_sum)
    plt.xlabel('X-coordinate value')
    plt.ylabel('Sum of pixel values over columns')
    plt.show()
    plt.plot(range(len(row_sum)),row_sum)
    plt.xlabel('Y-coordinate value')
    plt.ylabel('Sum of pixel values over rows')
    plt.show()
    return column_sum,row_sum

# To determine the peak values in the row and column sum and thus sample
# location.
def peak_values(column_sum,row_sum,n_columns,n_rows,image):
    ''' Function to find peaks from the column_sum and row_sum arrays
        and return a dataframe with sample locations.
        Args:
        column_sum: Sum of pixel values over all the columns in the
        image array.
        row_sum: Sum of pixel values over all the rows in the
        image array.
        n_columns: Number of columns of samples in the image
        n_rows: Number of rows of samples in the image.
        image: Image to be processed
        Returns: 
        sample_location: A dataframe containing sample and plate locations and a plot with locations
        superimposed on the image to be processed.
        
    '''
    column_peaks = find_peaks(column_sum,distance=10)
    column_peaks = column_peaks[0]
    row_peaks = find_peaks(row_sum,distance=10)
    row_peaks = row_peaks[0]
    X = []
    Y = []
    plate_location = []
    i = 0
    j = 0
    for i in range(0,n_rows):
        for j in range(0,n_columns):
            Y.append(column_peaks[j])
            if j == 0:
                plate_location.append(int((Y[j]-0)/2))
            else:
                plate_location.append(int((Y[j] + Y[j-1])/2))
            j = j + 1
            X.append(row_peaks[i])
            
        i = i + 1
    
    sample_location = pd.DataFrame(list(zip(X, Y, plate_location)),columns =['X', 'Y','plate_location'])
    plt.imshow(image)
    plt.scatter(sample_location['Y'],sample_location['X'],s=4)
    plt.scatter(sample_location['plate_location'],sample_location['X'],s=4)
    plt.show()
    return sample_location

# To determine the samle and plate temperature using peak locations.
def pixel_intensity(sample_location, frames, x_name, y_name, plate_name):
    ''' Function to find pixel intensity at all sample locations
        and plate locations in each frame.
        Args:
        sample_location(dataframe): A dataframe containing sample and plate locations.
        frames(list or array): An array of arrays containing all the frames of a video.
        x_name(string): Name of the column in sample_location containing the row values of the samples.
        y_name(string): Name of the column in sample_location containing the column values of the samples.
        plate_name(string): Name of the column in sample_location containing the column values of the
        plate location.
    '''
    temp = []
    plate_temp = []
    x = sample_location[x_name]
    y = sample_location[y_name]
    p = sample_location[plate_name]
    for i in range(len(sample_location)):
        temp_well = []
        plate_well_temp = []
        for frame in frames:
            temp_well.append(centikelvin_to_celsius(frame[x[i]][y[i]]))
            plate_well_temp.append(centikelvin_to_celsius(frame[x[i]][p[i]]))
        temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return temp,plate_temp

