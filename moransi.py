import numpy as np
#from skimage.io import imread
from libpysal.weights import lat2W
import pandas as pd
from esda.moran import Moran
from skimage.color import rgb2gray
from splot.esda import moran_scatterplot
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
Useful Links:
- https://blogs.oregonstate.edu/geo599spatialstatistics/2016/06/08/spatial-autocorrelation-morans/#:~:text=The%20Moran's%20I%20index%20will,Random%20is%20close%20to%20zero.
- https://pysal.org/esda/generated/esda.Moran.html
- https://onlinelibrary.wiley.com/doi/10.1111/j.1538-4632.2007.00708.x
- https://www.statology.org/morans-i/
- https://github.com/yatshunlee/spatial_autocorrelation
- https://github.com/yatshunlee/spatial_autocorrelation/blob/main/example/Spatial%20Autocorrelation.ipynb
'''

# Import data from csv
def get_data(file_name = "brain-glucose-KD-M1-isotopolouges.csv", dir = "/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data", keep_coord = False):
    '''
    Convert file from csv to dataframe and remove unnecessary columns 

    Parameters:
        - file_name: name of the file
        - dir: directory containing the file (exclude trailing forward slash)
        - keep_coord: flag indicating whether or not to include columns containing pixel coordinates (x and y)
    
    Returns:
        - data: dataframe of the data
    '''
    data_path = f'{dir}/{file_name}'
    data = pd.read_csv(data_path)
    if keep_coord:
        data = data.drop(labels = ['Unnamed: 0'], axis = 1)
    else:
        data = data.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    return data 


def data_to_image(brain_data, iso_num):
    '''
    Converts df of imaging data to an image by mapping the xy coordinates of the pixels. 

    Parameters:
        - brain_data: df (observations x metabolites) that has two columns labeled 'x' and 'y' to pull coordinates from
        - iso_num: the column index of the metabolite you wish to map to an image

    Returns: 
        - brain: an np array that can be plotted as an image
    '''
    # Setting x and y boundaries for individual plots, shifted down
    x_min = brain_data['x'].min()
    x_max = brain_data['x'].max()
    y_min = brain_data['y'].min()
    y_max = brain_data['y'].max()

    brain_data['x'] = brain_data['x'] - x_min
    brain_data['y'] = brain_data['y'] - y_min
    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    iso_to_plot = list(brain_data.keys())[iso_num]
    plotting_df = brain_data[[iso_to_plot, 'x', 'y']]

    brain = np.zeros((x_range,y_range)) 
    for index, row in plotting_df.iterrows():
        brain[int(row['x']), int(row['y'])] = row[iso_to_plot]

    return brain


def Morans_I(data, reshape = False, caption = "", plot = True):
    """
    Calculate the Moran's I score for an image and plot a scatterplot representation

    Parameters:
        - data: the image (as an np array)
        - reshape: (Bool) flag to reshape the image to grayscale if it starts as RGB
            - default: False
        - caption: (String) text to print alongside the Moran's i value
        - plot: (Bool) flag indicating whether or not to display the scatterplot
            - default: False 

    Returns:
        - round(MoranM.I,4): the Moran's I score of the image rounded to 4 decimal places 
    """

    # Transforming RGB data to grayscale
    if reshape:
        data_gray = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        data_gray = data
    col,row = data_gray.shape[:2]
    WeightMatrix= lat2W(row,col)
    WeightMatrix = lat2W(data_gray.shape[0],data_gray.shape[1])
    MoranM = Moran(data_gray,WeightMatrix)

    if plot:
        print(f"{caption}Moran's I Value:\t" +str(round(MoranM.I,4)))
        print("Raster Dimensions:\t" + str(data_gray.shape))

        fig, ax = moran_scatterplot(MoranM, aspect_equal=True)
        plt.show()

    return round(MoranM.I,4)


def testing_Morans_I():
    """
    Load in example images to test the Morans_I functionality. The example images are chess boards. 

    This function is NOT required for use with the Spatial MSI data.
    """
    raster_image_random = imread(r'./test-images/random.tif')
    raster_image_chess = imread(r'./test-images/chess.tif')
    raster_image_half = imread(r'./test-images/half+half_100.tif')

    fig1 = plt.imshow(raster_image_random)
    plt.title("Random Distribution\nexpected Moran's I: 0")
    plt.figure()
    plt.show()
    
    fig2 = plt.imshow(raster_image_chess)
    plt.title("Chessboard Pattern\nexpected Moran's I: -1")
    plt.figure()
    plt.show()
    
    fig3 = plt.imshow(raster_image_half)
    plt.title("Half&Half\nexpected Moran's I: 1")
    plt.figure()
    plt.show()

    Morans_I(raster_image_random, plot = True)
    Morans_I(raster_image_half, plot = True)


if __name__ == "__main__":

    '''
    Go through the list of datafiles provided, and for every file generate a list of valid metabolites (those above the cutoff Moran's I score)
    to keep and a list of those to throw away. Both lists will be written to a txt file. 

    txt_file: the path to the txt file that will contain the isotopolouge indices for valid and invalid isolouges as separate lists
    paths: list containing the filepaths for your data
    '''
    tracer_dir1 = 'Brain-15NGln'
    tracer1 = 'B15NGln'

    tracer_dir2 = 'Brain-15NLeu'
    tracer2 = 'B15NLeu'

    tracer = '3HB'

    txt_file = './Morans-values/brain-15NGln.txt'
    path1 = f'brain-m0-no-log/{tracer_dir1}/{tracer1}-KD-M1'
    path2 = f'brain-m0-no-log/{tracer_dir1}/{tracer1}-KD-M2'
    path3 = f'brain-m0-no-log/{tracer_dir1}/{tracer1}-KD-M3'
    path4 = f'brain-m0-no-log/{tracer_dir1}/{tracer1}-ND-M1'
    path5 = f'brain-m0-no-log/{tracer_dir1}/{tracer1}-ND-M2'
    path6 = f'brain-m0-no-log/{tracer_dir1}/{tracer1}-ND-M3'

    paths = [path1, path2, path3, path4, path5, path6]

    tracers = ["glucose", "lactate", "glutamine", "glycerol", "citrate", "3HB", "acetate"]
    replicates = ["M1", "M2", "M3"]

    '''
    # for filename in paths[0:3]:
    for tracer in tracers[1:]:
        for replicate in replicates:
            filename = f'kidney-m0-no-log/{tracer}-{replicate}'
    '''
    for filename in paths[3:]:
            # Print the file you are currently working on
            print(filename + '-FML')
            # Import the data - be sure to check the filename here
            testing_ions = False

            if testing_ions:
                data = get_data(file_name=f'{filename}-FML-ioncounts-ranks.csv', keep_coord=True)

            else:
                data = get_data(file_name=f'{filename}-FML-isotopolouges-ranks.csv', keep_coord=True)

            # Lists to hold the indices and names of valid and invalid isolouges
            isotopolouges_names = list(data.keys())
            valid_isotopolouges = []
            valid_isotopolouges_indices = []
            invalid_isotopolouges = []
            invalid_isotopolouges_indices = []
            morans_vals = []
            
            # Subtract 2 to account for the x and y coordinates at end of df
            print(len(isotopolouges_names) - 2) 
            for i in tqdm(range(len(isotopolouges_names) - 2)):
                # Print status every 10 metabolites
                #if i % 50 == 0:
                #    print(i)
                brain = data_to_image(data, i)
                morans_i = Morans_I(brain, reshape = False, caption = f'{isotopolouges_names[i]} ', plot=False)
                
                # The cutoff value of which metabolites to keep can be changed here
                if morans_i >= 0.6:
                    valid_isotopolouges.append(isotopolouges_names[i])
                    valid_isotopolouges_indices.append(i)
                else:
                    invalid_isotopolouges.append(isotopolouges_names[i])
                    invalid_isotopolouges_indices.append(i)

                morans_vals.append(morans_i)

            # Write to the txt file!
            f = open(txt_file, "a")
            f.write(f'{filename}-FML\n')
            f.write(f'Metab Names:\n')
            f.writelines(f'{str(isotopolouges_names)}\n')            
            f.write(f'Morans Vals:\n')
            f.writelines(f'{str(morans_vals)}\n')
            f.write(f'Valid Isotopolouges:\n')
            f.writelines(f'{str(valid_isotopolouges_indices)}\n')
            f.write(f'Invalid Isotopolouges:\n')
            f.writelines(f'{str(invalid_isotopolouges_indices)}\n')
            f.close()

            print(valid_isotopolouges_indices)
            print(invalid_isotopolouges_indices)

# source ./venv/bin/activate