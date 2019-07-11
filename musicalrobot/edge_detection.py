import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        regprops[i].sort_values(['Column','Row'],inplace=True)
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

#### Wrapping function ######
def centroid_temp(frames,n_samples):
    ''' Function to obtain sample temperature and plate temperature
        in every frame of the video using edge detection.
    Args:
        frames(List): An list containing an array for each frame
        in the video or just a single array in case of an image.
        n_samples: Number of samples in the video
    Returns:
        s_temp(List): A list containing a list a temperatures for each sample
        in every frame of the video 
        plate_temp(List): A list containing a list a temperatures for each plate
        location in every frame of the video.
    '''
    # Use the function 'flip_frame' to flip the frames horizontally 
    #and vertically to correct for the mirroring during recording
    flip_frames = flip_frame(frames)
    # Use the function 'edge_detection' to detect edges, fill and 
    # label the samples.
    labeled_samples = edge_detection(flip_frames)
    # Use the function 'regprop' to determine centroids of all the samples
    regprops = regprop(labeled_samples,flip_frames,n_samples)
    # Use the function 'sample_temp' to obtain temperature of samples 
    # and plate temp
    s_temp, plate_temp = sample_temp(regprops,flip_frames)
    return flip_frames, regprops, s_temp, plate_temp
