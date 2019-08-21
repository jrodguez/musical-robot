import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import skimage
import numpy as np
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
from scipy.interpolate import BSpline
from irtemp import centikelvin_to_celsius

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
    edges = feature.canny(frames[0]/1400)
    filled_samples = binary_fill_holes(edges)
    cl_samples = remove_small_objects(filled_samples,min_size = 15)
    labeled_samples = label(cl_samples)
    return labeled_samples

# Function to determine centroids of all the samples
def regprop(labeled_samples,frames,n_samples,n_rows,n_columns):
    ''' Determines the area and centroid of all samples.
        Args:
        labeled_samples(array): An array with labeled samples.
        flip_frames (array) : Original intensity image to determine
        the intensity at sample centroids.
        n_samples: Number of samples in the video being analyzed.
        n_rows: Number of rows of sample
        n_columns: Number of columns of sample
        Returns:
        A dictionary of dataframe with information about samples in every
        frame of the video.
    '''
    regprops = {} 
    for i in range(len(frames)):
        props = regionprops(labeled_samples, intensity_image=frames[i])
        row = np.zeros(len(props)).astype(int)
        column = np.zeros(len(props)).astype(int)
        area = np.zeros(len(props))
        perim = np.zeros(len(props))
        intensity = np.zeros(len(props),dtype=np.float64)
        plate = np.zeros(len(props),dtype=np.float64)
        plate_coord = np.zeros(len(props))
       
        c = 0
        for prop in props:
            row[c] = int(prop.centroid[0])
            column[c] = int(prop.centroid[1])
            #print(y[c])
            area[c] = prop.area
            perim[c] = prop.perimeter
            intensity[c] = frames[i][row[c]][column[c]]
            plate[c] = frames[i][row[c]][column[c]+10]
            plate_coord[c] = column[c]+10
            c = c + 1
         
        regprops[i] = pd.DataFrame({'Row': row, 'Column': column,'Plate':plate,'Plate_coord':plate_coord ,'Area': area,
                                'Perim': perim, 'Mean Intensity': intensity},dtype=np.float64)
        if len(regprops[i]) != n_samples:
            print('Wrong number of samples are being detected in frame %d' %i)    
        regprops[i].sort_values(['Column','Row'],inplace=True)
        # After sorting the dataframe according by columns in ascending order.
        # for j in range(0,n_columns):
        #     regprops[i][j*n_rows:(j+1)*n_rows].sort_values(['Row'],inplace=True)
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

# Function to obtain melting point by extracting the inflection point
def inflection_point(s_temp,p_temp):
    '''Function to determine inflection point in the sample temperature
    profile(melting point)
    Args:
    s_temp(list): Temperature of all the samples in every frame of the video.
    p_temp(list): Temperature of the plate next to every sample in every
        frame of the video.
    Returns:
    inf_temp(list): List of temperature at inflection points for each sample
    '''
    s_infl = []
    p_infl = []
    s_peaks = []
    p_peaks = []
    inf_peak = [] ; inf_temp = []
    for temp in s_temp:
        frames = np.linspace(1,len(temp),len(temp))
        # Fitting a spline to the temperature profile of the samples.
        bspl = BSpline(frames,temp,k=3)
        # Stacking x and y to calculate gradient.
        gradient_array = np.column_stack((frames,bspl(frames)))
        # Calculating gradient
        gradient = np.gradient(gradient_array,axis=0)
        # Calculating derivative
        derivative = gradient[:,1]/gradient[:,0]
        # Finding peaks in the derivative plot.
        peaks, properties = find_peaks(derivative,height=0.1)
        max_height1 = np.max(properties['peak_heights'])
        # To find the second highest peak
        a = list(properties['peak_heights'])
        a.remove(max_height1)
        max_height2 = np.max(a)
        # Appending the index of the two highest peaks to lists.
        inf_index1 = list(properties['peak_heights']).index(max_height1)
        inf_index2 = list(properties['peak_heights']).index(max_height2)
        # Appending the frame number in which these peaks occur to a list
        s_peaks.append([peaks[inf_index1],peaks[inf_index2]])
        # Appending the temperature at the peaks.
        s_infl.append([temp[peaks[inf_index1]],temp[peaks[inf_index2]]])
    for temp in p_temp:
        frames = np.linspace(1,len(temp),len(temp))
        bspl = BSpline(frames,temp,k=3)
        gradient_array = np.column_stack((frames,bspl(frames)))
        gradient = np.gradient(gradient_array,axis=0)
        derivative = gradient[:,1]/gradient[:,0]
        peaks, properties = find_peaks(derivative,height=0.1)
        max_height1 = np.max(properties['peak_heights'])
        # To find the second highest peak
        a = list(properties['peak_heights'])
        a.remove(max_height1)
        max_height2 = np.max(a)
        inf_index1 = list(properties['peak_heights']).index(max_height1)
        inf_index2 = list(properties['peak_heights']).index(max_height2)
        p_peaks.append([peaks[inf_index1],peaks[inf_index2]])
        p_infl.append([temp[peaks[inf_index1]],temp[peaks[inf_index2]]])
    for i,peaks in enumerate(s_peaks):
        for peak in peaks:
            if abs(peak - p_peaks[i][0]) >= 3:
                inf_peak.append(peak)
                break
            else:
                pass
    for i,temp in enumerate(s_temp):
        inf_temp.append(temp[inf_peak[i]])
    return inf_temp


#### Wrapping functions ######
# Wrapping function to get the inflection point
def inflection_temp(frames,n_samples,n_rows,n_columns):
    ''' Function to obtain sample temperature and plate temperature
        in every frame of the video using edge detection.
    Args:
        frames(List): An list containing an array for each frame
        in the video or just a single array in case of an image.
        n_samples: Number of samples in the video
        n_rows: Number of rows of sample
        n_columns: Number of columns of sample
        Returns:
        flip_frames(array) : An array of images which are flipped to correct the
        rotation caused by the IR camera
        regprops(dictionary) : A dictionary of dataframes containing temperature data.
        s_temp(List): A list containing a list a temperatures for each sample
        in every frame of the video 
        plate_temp(List): A list containing a list a temperatures for each plate
        location in every frame of the video.
        inf_temp: A list containing melting point of all the samples obtained by the plot.

    '''
    # Use the function 'flip_frame' to flip the frames horizontally 
    #and vertically to correct for the mirroring during recording
    flip_frames = flip_frame(frames)
    # Use the function 'edge_detection' to detect edges, fill and 
    # label the samples.
    labeled_samples = edge_detection(flip_frames)
    # Use the function 'regprop' to determine centroids of all the samples
    regprops = regprop(labeled_samples,flip_frames,n_samples,n_rows,n_columns)
    # Use the function 'sample_temp' to obtain temperature of samples 
    # and plate temp
    s_temp, p_temp = sample_temp(regprops,flip_frames)
    # Use the function 'infection_point' to obtain melting point of samples
    inf_temp = inflection_point(s_temp,p_temp)
    return flip_frames, regprops, s_temp, p_temp, inf_temp
