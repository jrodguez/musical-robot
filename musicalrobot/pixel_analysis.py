import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import skimage
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edge_detection

from skimage import io
from skimage import feature
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects  
from scipy.signal import find_peaks
from irtemp import centikelvin_to_celsius
from edge_detection import inflection_point

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
        frame(array):Equalized image 
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
    plt.xlabel('Column index')
    plt.ylabel('Sum of pixel values over columns')
    plt.title('Sum of pixel values over columns against column index')
    plt.show()
    plt.plot(range(len(row_sum)),row_sum)
    plt.xlabel('Row index')
    plt.ylabel('Sum of pixel values over rows')
    plt.title('Sum of pixel values over rows against row index')
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
    row = []
    column = []
    plate_location = []
    i = 0
    j = 0
    for i in range(0,n_columns):
        for j in range(0,n_rows):
            row.append(row_peaks[j])
            column.append(column_peaks[i])
            if j == 0:
                plate_location.append(int((row[j]-0)/2))
            else:
                plate_location.append(int((row[j] + row[j-1])/2))
            j = j + 1
        i = i + 1
    
    sample_location = pd.DataFrame(list(zip(row, column, plate_location)),columns =['Row', 'Column','plate_location'])
    plt.imshow(image)
    plt.scatter(sample_location['Column'],sample_location['Row'],s=4)
    plt.scatter(sample_location['Column'],sample_location['plate_location'],s=4)
    plt.title('Sample and plate location at which the temperature profile is monitored')
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
            plate_well_temp.append(centikelvin_to_celsius(frame[p[i]][y[i]]))
        temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return temp,plate_temp


##### Wrapping Function ######
def pixel_temp(frames,n_frames,n_columns,n_rows):
    ''' Function to determine the temperature of the samples and plate locations by analysing 
    pixel values and finding peaks.
    Args:
    frames: The frames of a video to be analysed.
    n_frames: Number of frames in the video
    n_columns: Number of columns of samples in the image
    n_rows: Number of rows of samples in the image.
    Returns:
    m_df(Dataframe): A dataframe containing row and column coordinates of each sample 
    and its respective inflection point obtained. 
    '''
    flip_frames = edge_detection.flip_frame(frames)
    #Function to obtained an equalized image using all the frames
    #in the video.
    img_eq = image_eq(n_frames,flip_frames)
    #Funtion to determine sum of pixels over all the rows and columns
    #to obtain plots with peaks at the sample position in the array.
    column_sum,row_sum = pixel_sum(img_eq)
    # Function to find peaks from the column_sum and row_sum arrays
    # and return a dataframe with sample locations.
    sample_location = peak_values(column_sum,row_sum,n_columns,n_rows,img_eq)
    # Function to find pixel intensity at all sample locations
    # and plate locations in each frame.
    temp,plate_temp = pixel_intensity(sample_location, frames, x_name = 'Row', y_name = 'Column', plate_name = 'plate_location')
    # Function to obtain the inflection point(melting point) from the temperature profile.
    inf_temp, s_peaks, p_peaks = inflection_point(temp,plate_temp)
    # Dataframe with sample location (row and column coordinates) and respective inflection point.
    m_df = pd.DataFrame({'Row':sample_location.Row,'Column':sample_location.Column,'Melting point':inf_temp})
    return m_df

