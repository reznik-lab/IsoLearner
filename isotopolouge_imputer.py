import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Add, Dense, Activation, BatchNormalization, Lambda, ReLU, PReLU, LayerNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display
import seaborn as sns
from scipy import stats
from visualization import *
from collections import Counter
from statsmodels.stats.multitest import multipletests

from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor # for building the model
from prettytable import PrettyTable
import os
import re

# Import data from csv
def get_data(file_name = "brain-glucose-KD-M1-isotopolouges.csv", dir = "/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data", keep_coord = False):
    '''
    Convert file from csv to dataframe and remove unnecessary columns 

    Parameters:
        - file_name: name of the file
        - dir: Absolute path to the directory containing the file (exclude trailing forward slash)
    
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

# Model
def multiple_regression_model(num_ion_counts, num_isotopolouges, lambda_val):
    model = Sequential([
        # Input Layer
        Dense(128, input_dim = num_ion_counts, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dropout(0.25),      
        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(num_isotopolouges, kernel_initializer='he_uniform', activation = 'relu', kernel_regularizer=l2(lambda_val))
    ])

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                #loss=tf.keras.losses.MeanSquaredError(),
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=['mse', 'mae'])

    return model

def multiple_regression_model_2(num_ion_counts, num_isotopolouges, lambda_val):
    # Leaky RELU
    model = Sequential([
        # Input Layer
        LayerNormalization.adapt(axis=1),
        Dense(128, input_dim = num_ion_counts, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dropout(0.25),      
        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        Dense(num_isotopolouges, kernel_initializer='he_uniform', kernel_regularizer=l2(lambda_val)),
        LayerNormalization(axis=1),
    ])

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                #loss=tf.keras.losses.MeanSquaredError(),
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=['mse', 'mae'])

    return model

def FML_regression_model(num_ion_counts, num_isotopolouges, lambda_val):
    model = Sequential([
        # Input Layer
        Dense(128, input_dim = num_ion_counts, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),

        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dropout(0.25),
        
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        
        Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dropout(0.25),      
        
        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        
        # Removed relu to allow negative 
        Dense(num_isotopolouges, kernel_initializer='he_uniform', kernel_regularizer=l2(lambda_val))
    ])

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 3e-05),
                #loss=tf.keras.losses.MeanSquaredError(),
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=['mse', 'mae'])

    return model

def create_large_data(all_data = True, data_path = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data', primary_name = 'brain-glucose-KD-M1-isotopolouges.csv'):
    '''
    Creates feature and target dataframes consisting of different samples concatenated to each other. Assumes there are 6 total - 3 KD and 3 ND. 

    params: 
        - all_data: Bool for whether or not all 6 samples should be turned into a single training file, or the last sample (ND-M3) should be omitted and instead returned as a
                    separate pair of train/test files for later use.
    '''
    # List containing the file names of the isotopolouge data
    isotopolouges_paths = [f'{data_path}/brain-glucose-KD-M{i+1}-isotopolouges.csv' for i in range(3)]
    isotopolouges_paths.extend([f'{data_path}/brain-glucose-ND-M{i+1}-isotopolouges.csv' for i in range(2)])

    # List containing the file names of the ion count data
    ion_counts_paths = [f'{data_path}/brain-glucose-KD-M{i+1}-ioncounts.csv' for i in range(3)]
    ion_counts_paths.extend([f'{data_path}/brain-glucose-ND-M{i+1}-ioncounts.csv' for i in range(2)])

    # If all_data flag activated, include the final brain sample, otherwise return a test feature and target
    if all_data:
        isotopolouges_paths.append(f'{data_path}/brain-glucose-ND-M3-isotopolouges.csv')
        ion_counts_paths.append(f'{data_path}/brain-glucose-ND-M3-ioncounts.csv')
    else:
        test_features = get_data(file_name = 'brain-glucose-ND-M3-ioncounts.csv')
        test_targets = get_data(file_name = 'brain-glucose-ND-M3-isotopolouges.csv')

    # Load each dataframe from the list and concatenate them to each other for - isotopolouges
    isotopolouges = pd.read_csv(isotopolouges_paths[0])
    for i, path in enumerate(isotopolouges_paths):
        if i == 0:
            continue 
        data = pd.read_csv(path)
        isotopolouges = pd.concat([isotopolouges, data], ignore_index=True, axis = 0)

    # Load each dataframe from the list and concatenate them to each other for - ions
    ion_counts = pd.read_csv(ion_counts_paths[0])
    for i, path in enumerate(ion_counts_paths):
        if i == 0:
            continue 
        data = pd.read_csv(path)
        ion_counts = pd.concat([ion_counts, data], ignore_index=True, axis = 0)

    # Drop the unneeded columns from both features and targets 
    ion_counts = ion_counts.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    isotopolouges = isotopolouges.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)

    if all_data:
        return ion_counts, isotopolouges
    else:
        return ion_counts, isotopolouges, test_features, test_targets 

def create_large_data_ranked(all_data = True, data_path = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data', FML = False):
    '''
    Creates feature and target dataframes consisting of different samples concatenated to each other. Assumes there are 6 total - 3 KD and 3 ND. 

    params: 
        - all_data: Bool for whether or not all 6 samples should be turned into a single training file, or the last sample (ND-M3) should be omitted and instead returned as a
                    separate pair of train/test files for later use.
    '''

    iso_path = 'FML-isotopolouges-ranks' if FML else 'isotopolouges-ranks'
    ion_path = 'FML-ioncounts-ranks' if FML else 'ioncounts-ranks'

    # List containing the file names of the isotopolouge data
    isotopolouges_paths = [f'{data_path}/BG-KD-M{i+1}-{iso_path}.csv' for i in range(3)]
    isotopolouges_paths.extend([f'{data_path}/BG-ND-M{i+1}-{iso_path}.csv' for i in range(2)])

    # List containing the file names of the ion count data
    ion_counts_paths = [f'{data_path}/BG-KD-M{i+1}-{ion_path}.csv' for i in range(3)]
    ion_counts_paths.extend([f'{data_path}/BG-ND-M{i+1}-{ion_path}.csv' for i in range(2)])

    # If all_data flag activated, include the final brain sample, otherwise return a test feature and target
    if all_data:
        isotopolouges_paths.append(f'{data_path}/BG-ND-M3-{iso_path}.csv')
        ion_counts_paths.append(f'{data_path}/BG-ND-M3-{ion_path}.csv')
    else:
        test_features = get_data(file_name = f'BG-ND-M3-{ion_path}.csv', dir = data_path)
        test_targets = get_data(file_name = f'BG-ND-M3-{iso_path}.csv', dir = data_path)

    # Load each dataframe from the list and concatenate them to each other for - isotopolouges
    isotopolouges = pd.read_csv(isotopolouges_paths[0])
    for i, path in enumerate(isotopolouges_paths):
        if i == 0:
            continue 
        data = pd.read_csv(path)
        isotopolouges = pd.concat([isotopolouges, data], ignore_index=True, axis = 0)

    # Load each dataframe from the list and concatenate them to each other for - ions
    ion_counts = pd.read_csv(ion_counts_paths[0])
    for i, path in enumerate(ion_counts_paths):
        if i == 0:
            continue 
        data = pd.read_csv(path)
        ion_counts = pd.concat([ion_counts, data], ignore_index=True, axis = 0)

    # Drop the unneeded columns from both features and targets 
    ion_counts = ion_counts.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    isotopolouges = isotopolouges.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)

    if all_data:
        return ion_counts, isotopolouges
    else:
        return ion_counts, isotopolouges, test_features, test_targets


# ***************************************************** CHECKED FOR GENERALIZABILITY *****************************************************

# ============================================== LIST OF FILEPATHS =====================================================================
def generate_filepath_list(data_path = '/brain-m0-no-log', FML = True, tracer = 'BG'):
    '''
    Returns relative paths of data files as two lists. If sample includes both normal and ketogenic replicates, the ND replicates are first, and then KD. 
        - Example Filename: 'B3HB-KD-M1-FML-ioncounts-ranks.csv'

    Parameters: 
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose

    Returns: 
        - ion_counts_paths (list): list of filenames with ion_count data
        - isotopologues_paths (list): list of filenames with iso data
    '''
    iso_path = 'FML-isotopolouges-ranks' if FML else 'isotopolouges-ranks'
    ion_path = 'FML-ioncounts-ranks' if FML else 'ioncounts-ranks'

    # List containing the file names of the isotopolouge data - normal diet mice
    isotopologues_paths = [f'{data_path}/{tracer}-ND-M{i+1}-{iso_path}.csv' for i in range(3)]
    # List containing the file names of the ion count data - normal diet mice
    ion_counts_paths = [f'{data_path}/{tracer}-ND-M{i+1}-{ion_path}.csv' for i in range(3)]

    # These two tracers have Ketogenic mice as well, include them in the filepaths
    if tracer == 'BG' or tracer == 'B3HB':
        isotopologues_paths.extend([f'{data_path}/{tracer}-KD-M{i+1}-{iso_path}.csv' for i in range(3)])
        ion_counts_paths.extend([f'{data_path}/{tracer}-KD-M{i+1}-{ion_path}.csv' for i in range(3)])

    return ion_counts_paths, isotopologues_paths

# ============================================== IDENTIFYING ION + ISO INCONSISTENCIES ============================================================
def checking_data_consistency(data_path = '/brain-m0-no-log', FML = True, tracer = 'BG'):
    '''
    Identifies inconsistent metabolites for both ion count and isotopologue data for given tracer set of replicates. 

    params: 
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose

    returns: 
        - ion_inconsistencies (list): list containing the names of the metabolites that are not common in all ion_count files.
        - iso_inconsistencies (list): list containing the name of the isotopologues that are not common in all iso files. 
    '''

    # Generate lists containing the paths to the ion count and isotopologue data
    ion_counts_paths, isotopolouges_paths = generate_filepath_list(data_path = data_path, FML = FML, tracer = tracer)

    # List of ions that need to be removed from all files
    ion_inconsistencies = identify_inconsistencies(ion_counts_paths, show_progress = False)
    # List of isotopolouges that need to be removed from all files
    iso_inconsistencies = identify_inconsistencies(isotopolouges_paths, show_progress = False)

    return ion_inconsistencies, iso_inconsistencies

# ============================================== IDENTIFYING DATA INCONSISTENCIES ============================================================
def identify_inconsistencies(list_of_paths, show_progress = True):
    '''
    Helper Function - Goes through multiple datafiles and identifies metabolites that do not appear in all files. 
    These metabolites/isotopolouges would then be removed prior to training the model. 

    Parameters:
        - list_of_paths (list): list containing the relative file paths for csvs that need to be compared

    Returns: 
        - invalid_metabs_names (list): list containing the names (not indices) of the metabolites that are not common in all files.
    '''

    # Number of replicates
    num_replicates = len(list_of_paths)
    # Holds all metabolites of all files (including duplicates)
    individual_replicate_metabs = []
    # List of lists, where each sublist is the metabolites for a single file
    all_metabs = []

    for i, name in enumerate(list_of_paths):
        # Load data
        ion_count = get_data(file_name = name, dir = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data')
        metab_names = ion_count.columns

        if show_progress:
            print(i, name, len(metab_names))

        individual_replicate_metabs.append(metab_names)
        all_metabs.extend(metab_names)

    # Flatten the list of lists into single lists
    all_metabs.sort()

    # Returns a dictionary where the keys are the iso indices and the values are the number of times they appear in the flattened list (a count)
    metab_index_dict = Counter(all_metabs)
    # Create a list the names of all metabolites that do not appear in all replicates
    invalid_metabs_names = [index for index in list(metab_index_dict.keys()) if metab_index_dict[index] < num_replicates]

    return invalid_metabs_names
   
# ============================================== REMOVING DATA INCONSISTENCIES ============================================================
def remove_data_inconsistencies(additional_ion_metabolite_to_remove = [], additional_metabs_to_remove = [], data_path = '/brain-m0-no-log', FML = True, tracer = 'BG'):
    '''
    Generates the final dataset to use for regression. Loads in the relevant input files, and then removes two different sets of metabolites from each replicate:
        1). Metabolites/isotopologues that are not consistent across replicates (were not detected through mass spec for some replicates)
        2). Metabolites/isotopologues that were deemed invalid by failing to surpass the Moran's I metric for the majority of replicates

    Paremeters:
        - additional_metabs_to_remove (list): list of isotopologue NAMES (not indices) that must be removed from all replicates (ie the names from Moran's)
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose

    Returns:
        - clean_ion_data (list): list of ion count dataframes that are n (number of pixels in this replicate - can be different for each) by m (num of metabolites - consistent across all)
        - clean_iso_data (list): list of isotopologue dataframes that are n (number of pixels in this replicate - can be different for each) by m (num of isotopologues for prediction - consistent across all)
    '''

    # Lists of filepaths to ion_counts and isotopolouges
    ion_counts_paths, isotopolouges_paths = generate_filepath_list(data_path = data_path, FML = FML, tracer = tracer)
    # Lists of names of inconsistent  metabolites and isotopolouges that need to be removed
    ion_inconsistencies, iso_inconsistencies = checking_data_consistency(data_path = data_path, FML = FML, tracer = tracer)

    print(f"Inconsistencies found: {len(ion_inconsistencies)} metabolites, {len(iso_inconsistencies)} isotopolouges")

    # Append Moran's metabs list to iso list for removal
    iso_inconsistencies += additional_metabs_to_remove
    # Remove any duplicate names
    iso_inconsistencies = [*set(iso_inconsistencies)]
    print(f"Removing {len(iso_inconsistencies)} isotopolouges")

    clean_ion_data = []
    clean_iso_data = []

    for i, data_path in enumerate(ion_counts_paths):
        # Load in the data for single replicate
        data = get_data(file_name = data_path, dir = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data')
        # Get list of metabolites for that replicate
        metabolite_names = data.columns
        # List of metabolites that must be dropped that are present in this replicate (this step actually not necessary) 
        metab_to_drop = [metab for metab in ion_inconsistencies if metab in metabolite_names]

        # Drop the unneeded metabolites
        data = data.drop(labels = metab_to_drop, axis = 1)
        new_metabolite_names = data.columns

        print(f"File {i}: {data_path} || {len(metab_to_drop)} to drop || {len(metabolite_names) - len(new_metabolite_names)} dropped")

        # Append to list of cleaned/filtered dataframes for iso data
        clean_ion_data.append(data)

    # Confirm that all the dataframes have the same columns in the same order!
    checks = [True if (list(item.columns) == list(clean_ion_data[0].columns)) else False for item in clean_ion_data[1:]]
    if all(checks):
        print("Ion-Data is all consistent! Time to train a model!")
    else:
        print("THERE HAS BEEN AN ERROR!!!! Dataframes columns not all the same order.")

    # same thing for isolouges
    for i, data_path in enumerate(isotopolouges_paths):
        data = get_data(file_name = data_path, dir = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data')
        metabolite_names = data.columns
        metab_to_drop = [metab for metab in iso_inconsistencies if metab in metabolite_names]

        data = data.drop(labels = metab_to_drop, axis = 1)
        new_metabolite_names = data.columns

        print(f"File {i}: {data_path} || {len(metab_to_drop)} to drop || {len(metabolite_names) - len(new_metabolite_names)} dropped")

        clean_iso_data.append(data)

    # Confirm that all the dataframes have the same columns in the same order!
    checks = [True if (list(item.columns) == list(clean_iso_data[0].columns)) else False for item in clean_iso_data[1:]]
    if all(checks):
        print("Iso-Data is all consistent! Time to train a model!")
    else:
        print("THERE HAS BEEN AN ERROR!!!! Dataframes columns not all the same order.")

    return clean_ion_data, clean_iso_data

# ============================================== Creating Dataset for Training  ===========================================================
def create_full_dataset(ion_dfs, iso_dfs, holdout = True, holdout_index = 0):
    '''
    Take in the list of cleaned/consistent dataframes and returns a training a test set

    Parameters: 
        - ion_dfs (list): list of dataframes where each element is a replicate's ion_count data
        - iso_dfs (list): list of dataframes where each element is a replicate's isotopologue data
        - holdout (bool): flag indicating whether there is a replicate being held out for testing or all replicates should be used for training
        - holdout_index (int): The index in the ion and iso lists of the replicate that should be held out as the testing set
    
    Returns: 
        - ion_counts (df): # pixels x # metabolites df of ion_count data. Consists of multiple replicates for training set. 
        - isotopolouges (df): # pixels x # isotopologues df of isotopologue data. Consists of multiple replicates for training set. 
        - test_ion_counts (df): single holdout replicates df # pixels x # metabolites df of ion_count data for test set.  
        - test_iso_counts (df): single holdout replicates df # pixels x # isotopologues df of ion_count data for test set.  
    '''
    if holdout:
        test_ion_counts = ion_dfs.pop(holdout_index)
        test_iso_counts = iso_dfs.pop(holdout_index)

        stop_index = len(ion_dfs) - 1

        ion_counts = ion_dfs[0]
        for i, data in enumerate(ion_dfs[1:stop_index]):
            ion_counts = pd.concat([ion_counts, data], ignore_index = True, axis = 0)

        isotopolouges = iso_dfs[0]
        for i, data in enumerate(iso_dfs[1:stop_index]):
            isotopolouges = pd.concat([isotopolouges, data], ignore_index = True, axis = 0)

        return ion_counts, isotopolouges, test_ion_counts, test_iso_counts

    else:
        ion_counts = ion_dfs[0]
        for i, data in enumerate(ion_dfs[1:]):
            ion_counts = pd.concat([ion_counts, data], ignore_index = True, axis = 0)
        
        isotopolouges = iso_dfs[0]
        for i, data in enumerate(iso_dfs[1:]):
            isotopolouges = pd.concat([isotopolouges, data], ignore_index = True, axis = 0)

        return ion_counts, isotopolouges

# ***************************************************** CHECKED FOR GENERALIZABILITY *****************************************************


# ============================================== PROCESSING MORANS I  =====================================================================
def indices_to_metab_names(list_of_morans_metabs, metabs_to_consider = 'isos', data_path = '/brain-m0-no-log', tracer = 'BG', FML = True, cutoff = 4):
    '''
    This function iterates through the list of metabolite indices (identified by the Moran's I score), and returns a list 
    of the metabolite names that should be removed from consideration based on the number of replicates they are poor targets in. 

    Moran's I was run on each file individually, so the indices for each file may not correspond to each other as intended. Instead,
    this function will convert each list of indices to the isotopolouge name of that specific file, and then the comparisons will
    be done on the isotopolouge names rather than indices. 

    Add this list of metab names to the list of inconsistencies and remove them all from every file. 

    parameters: 
        - list_of_morans_metabs (list): list of lists, where every sublist contains the indices of metabolites that are poor targets in
            a single replicate. Each sublist corresponds to a single replicate.

        - metabs_to_consider (string, default = 'isos'): flag that determines whether the function should remove these metabolites from the 
            ion_count files or isotopologue files.

        - data_path (string): file path to the directory containing the relevant isotopologue files. 

        - FML (bool): flag indicating whether to use the full metabolite list or partial metabolite list.

        - cutoff (int, default = 4): Number corresponding to how many replicates should identify a metabolite as invalid before it is marked for removal.

    returns: 
        - invalid_metabs (list): a list of metabolite names that were marked as poor targets for prediction in at least (cutoff) replicates.
    '''

    # Paths to files
    ion_counts_paths, isotopolouges_paths = generate_filepath_list(data_path = data_path, FML = FML, tracer = tracer)
    # Iterate through either ion files or iso files 
    paths_to_iterate = isotopolouges_paths if metabs_to_consider == 'isos'  else ion_counts_paths

    all_metabs = []

    for i, data_tuple in enumerate(zip(paths_to_iterate, list_of_morans_metabs)):
        data = get_data(file_name = data_tuple[0], dir = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data')
        metab_names = list(data.columns)
        print(data_tuple[0], data_tuple[1][-1], len(metab_names))
        morans_metabs_names = [metab_names[index] for index in data_tuple[1] if index < len(metab_names)]
        all_metabs.append(morans_metabs_names)
        # print(len(morans_metabs_names))

    invalid_metabs = count_list_elements(all_metabs, cutoff=cutoff)

    return invalid_metabs


# ============================================== Training  ===========================================================
def preprocess_data(ion_counts, isotopolouges, testing_split = True):
    '''
    Take in the input dataframes and convert the to numpys for the model
    Params: 
        - ion_counts: dataframe containing the feature data
        - isotopologues: dataframe containing the target data
        - testing_split: flag indicating whether to produce a testing set or only train/val split
    Returns: 
        - num_ion_counts: the number of features going into the model
        - num_isotopolouges: the number of targets being predicted
    '''
    x = ion_counts.to_numpy()
    y = isotopolouges.to_numpy()
    # print(x.shape, y.shape)
    num_ion_counts = x.shape[1]
    num_isotopolouges = y.shape[1] if len(y.shape) != 1 else 1

    if testing_split:
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)
        return num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val, x_test, y_test
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        return num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val

def training(feature_data, target_data, checkpoint_path = './saved-weights/KD-M1-unranked-dropout/checkpoint', train = True, TRAIN_ENTIRE_BRAIN = False, ranked = True, EPOCHS = 100, BATCH_SIZE = 32):
    print("HI")
    # Whether or not to treat the dataset as training data
    if TRAIN_ENTIRE_BRAIN:
        num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val = preprocess_data(feature_data, target_data, testing_split = False)
    else:
        num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(feature_data, target_data)
    
    isotopolouge_names = list(target_data.columns)
    # print(isotopolouge_names)

    # define model
    model = FML_regression_model(num_ion_counts, num_isotopolouges, 0.01)

    # Checkpoints
    # https://keras.io/api/callbacks/model_checkpoint/
    checkpoint_filepath = checkpoint_path # './saved-weights/KD-M1-unranked-dropout/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose = 1,
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_best_only=False,
        save_freq= int(664 * 10))

    # Learning rate scheduler 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose = 1, min_lr=0)

    # fit model
    if train:
        history = model.fit(x_train, y_train, batch_size = BATCH_SIZE, verbose=1, validation_data = (x_val, y_val), epochs=EPOCHS, callbacks=[model_checkpoint_callback])
    
    else:
        model.load_weights(checkpoint_filepath)
        # evaluate model on test set
        mae = model.evaluate(x_test, y_test, verbose=1)

        prediction = model.predict(x_test)
        #plt.scatter(y_test, prediction)
        # plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-predicting.png')
        #plt.show()

        print(y_test.shape, prediction.shape)

        # print(y_test)
        # print(prediction)
        plot_individual_isotopolouges_2(y_test, prediction, isotopolouge_names, ranked = ranked)

    return history

def correlations():
    brain_K1 = get_data(file_name="brain-glucose-KD-M1-isotopolouges.csv")
    brain_K1_ranks = get_data(file_name="brain-glucose-KD-M1-isotopolouges-ranks.csv")
    
    brain_K2 = get_data(file_name="brain-glucose-KD-M2-isotopolouges.csv")
    brain_K2_ranks = get_data(file_name="brain-glucose-KD-M2-isotopolouges-ranks.csv")

    brain_N1 = get_data(file_name="brain-glucose-ND-M1-isotopolouges.csv")
    brain_N1_ranks = get_data(file_name="brain-glucose-ND-M1-isotopolouges-ranks.csv")

    #corr_heatmap(brain_1)
    #corr_heatmap(brain_1_ranks)

    double_corr_heatmap(brain_K1, brain_K1_ranks, title = "KD Brain 1 Pairwise Corr Coeff")
    double_corr_heatmap(brain_K2, brain_K2_ranks, title = "KD Brain 2 Pairwise Corr Coeff")
    double_corr_heatmap(brain_N1, brain_N1_ranks, title = "N1 Brain 1 Pairwise Corr Coeff")

    plt.show()
    isotopolouge_names = list(brain_K1.columns)

    corr_scatter(brain_K1, isotopolouge_names, 1, 2)

def test_whole_brain(feature_data, target_data, checkpoint_path = './saved-weights/KD-M1-unranked-dropout/checkpoint', ranked = False):
    isotopolouge_names = list(target_data.columns)
    num_ion_counts = feature_data.shape[1]
    num_isotopolouges = target_data.shape[1]

    features = feature_data.to_numpy()
    targets = target_data.to_numpy()

    # define model
    model = FML_regression_model(num_ion_counts, num_isotopolouges, 0.01)

    # Checkpoints
    # https://keras.io/api/callbacks/model_checkpoint/
    model.load_weights(checkpoint_path).expect_partial()
    
    prediction = model.predict(features)
    plt.scatter(targets, prediction)
    max = targets.max()
    min = targets.min()
    # Add the identity line 
    X_plot = np.linspace(min, max, 100)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color='r')
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")

    # plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-predicting.png')
    plt.show()
    
    print(targets.shape, prediction.shape)
    plot_individual_isotopolouges_2(targets, prediction, isotopolouge_names, grid_size = 5, ranked = ranked)

    '''
    # Making new csv
    df1 = pd.read_csv("/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M2-isotopolouges.csv")
    df2 = pd.DataFrame(prediction, columns = isotopolouge_names)
    df2['x'] = df1['x']
    df2['y'] = df1['y']
    # df2.to_csv('KDM2-predicted.csv')

    # print(df2)
    '''

def predict(feature_data, target_data, checkpoint_path = './saved-weights/KD-M1-unranked-dropout/checkpoint'):
    '''
    Predicts the isotopolouge breakdowns of a given list of metabolite, using the weights of the saved path. 

    Returns: 
        - prediction_df: a dataframe in which the columns are individual isotopolouges and the rows are observations
    '''
    isotopolouge_names = list(target_data.columns)
    num_ion_counts = feature_data.shape[1]
    num_isotopolouges = target_data.shape[1]

    features = feature_data.to_numpy()
    # targets = target_data.to_numpy()

    # define model
    model = FML_regression_model(num_ion_counts, num_isotopolouges, 0.01)

    # Checkpoints
    # https://keras.io/api/callbacks/model_checkpoint/
    checkpoint_filepath = checkpoint_path # './saved-weights/KD-M1-unranked-dropout/checkpoint'
    model.load_weights(checkpoint_filepath).expect_partial()
    
    prediction = model.predict(features)
    prediction_df = pd.DataFrame(prediction, columns = isotopolouge_names)

    return prediction_df

def spearman_rankings(actual, predicted, plot = True):
    '''
    For each pair of actual/predicted isotopolouges, calculates the Spearman's rank correlation coefficient. Allows us to know which isotopolouges are best predicted by this model.

    Paramters: 
        - actual (df): the ground truth dataframe for the regression model's predictions to be compared too. 
    '''
    spearmans = []
    p_values = []

    actual = actual.drop(labels = ['x', 'y'], axis = 1, errors = 'ignore')
    predicted = predicted.drop(labels = ['x', 'y'], axis = 1, errors = 'ignore')

    # Save the p values and then regulate them with this:
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html 
    for i in range(len(list(actual.columns))):
        spearman_coeff = stats.spearmanr(actual.iloc[:, i], predicted.iloc[:, i])
        spearmans.append(spearman_coeff.correlation)
        p_values.append(spearman_coeff.pvalue)

    # print(spearmans)        
    df_spearman = pd.DataFrame()
    df_spearman["median_rho"] = spearmans
    df_spearman["isotopologue"] = list(actual.columns)
    # Calculate q values and plot as color
    corrected_pvals = multipletests(p_values, method="fdr_bh", alpha = 0.1)
    df_spearman['pvals'] = corrected_pvals[1]
    color = ["purple" if pval <= 0.1 else "red" for pval in corrected_pvals[1]]
    df_spearman["color"] = color

    # Pull p-value 
    if plot:
        median_rho_feature_plot(df_spearman)

    sorted = df_spearman.sort_values(by=["median_rho"], ascending=False)

    return sorted

def mean_std_var_isotopolouge(isotopolouges, plot = False, mean_cutoff = 0.2):
    '''
    For given isotopolouge data, divides each individual isotopolouge by it's total metabolite ion count (the sum of the isotopolouges
    from the same metabolite). The mean and variance for each isotopolouge are then calculated and plotted. The isotopolouges are ranked
    from highest to lowest mean, and a list of the isotopologue names with a mean higher than a specific cut off is returned.  

    Parameters:
        - isotopolouges: a dataframe in which the the columns are isotopolouges 
            - (follows the naming convention [metab_01 m+00 | metab_01 m+01 | metab_01 m+02 ... metab_n m+05])

    Returns: the mean and standard deviation of each isotopolouge divided by the total ion count (sum of isotopolouges for that metabolite)
    '''
    
    pd.options.mode.chained_assignment = None   # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

    # List of all isotopolouges in the dataset  
    isotopolouge_names = list(isotopolouges.keys())
    isotoplouge_column_indices = []
    single_metabolite_indices = []
    reference_name = ""
    num_isotopolouges = len(isotopolouge_names)

    # Generates isotoplouge_column_indices -> contains sublists which each contain the indices of all the isotopolouges for a single metabolite
    # Ex -> [[0,1,2], [3, 4, 5], [6, 7, 8, 9 , 10 , 11]]
    # single_metabolite_indices -> the sublist containing all the indices for a single metabolite
    for i, isotopolouge_name in enumerate(isotopolouge_names):
        # Access the primary metabolite name, ignoring the isotopolouge m+0x
        metabolite_name = isotopolouge_name.split()[0]

        if reference_name == metabolite_name:
            single_metabolite_indices.append(i)
        else:
            isotoplouge_column_indices.append(single_metabolite_indices)
            single_metabolite_indices = []
            single_metabolite_indices.append(i)
            reference_name = metabolite_name
        
        if i == num_isotopolouges - 1:
            isotoplouge_column_indices.append(single_metabolite_indices)

    # Remove the initial empty sublist that was generated 
    isotoplouge_column_indices.pop(0)
    # print(isotoplouge_column_indices)

    # Instantialize dataframe that will hold all isotopolouges divided by the sum of their respective metabolite
    sum_df = pd.DataFrame()

    # For each metabolite, create a buffer df with it's isotopolouges, and calculate their sum. Divide them by this sum. 
    for i, sublist in enumerate(isotoplouge_column_indices):
        # Access the isotopolouges and take the sum
        temp_metab_df = isotopolouges.iloc[:, isotoplouge_column_indices[i]]
        temp_metab_df['sum'] = temp_metab_df.sum(axis = 'columns')
        
        # Move the sum to the first column (easier to do the division later)
        first_column = temp_metab_df.pop('sum')
        temp_metab_df.insert(0, 'sum', first_column)

        # DF where each isotopolouge is divided by the sum of all isotopolouges for that metabolite
        temp_metab_df.iloc[:,1:] = temp_metab_df.iloc[:,1:].div(temp_metab_df['sum'], axis=0)
        temp_metab_df.pop('sum')

        # Concatenate the new isotopolouge values to the final df to return 
        sum_df = pd.concat([sum_df, temp_metab_df], axis = 1, ignore_index=True)

    # Relabel the df with the isotopolouge names 
    sum_df.set_axis(isotopolouge_names, axis=1, inplace=True)
    # display(sum_df)

    # Calculate the means of each isotopolouge and sort from largest to smallest.
    means = sum_df.mean().sort_values(ascending = False)
    means_filtered = means[means >= mean_cutoff]

    # List of isotopolouge names that may be able to be predicted well (are above the mean cutoff)
    valid_isotopolouges = list(means_filtered.keys())
    valid_isotopolouges_indices = [isotopolouge_names.index(isotopolouge) for isotopolouge in valid_isotopolouges]
    valid_isotopolouges_indices.sort()

    valid_isotopolouges_ordered = [isotopolouge_names[index] for index in valid_isotopolouges_indices]

    stds = sum_df.std()
    if plot:
        means.plot.bar().axhline(y = mean_cutoff, color = "red")
        plt.show()
        stds.plot.bar()
        plt.show()

    # Extract the isotopolouge data that we want to use 
    # print(isotopolouges.iloc[:, valid_isotopolouges_indices])

    return [valid_isotopolouges_indices, valid_isotopolouges_ordered], isotopolouges.iloc[:, valid_isotopolouges_indices]

def df_concat(df1, df2):
    '''
    Concatenates two dataframes with the same number of rows side by side, returns the resulting dataframe
        - https://stackoverflow.com/questions/23891575/how-to-merge-two-dataframes-side-by-side
    '''
    return pd.concat([df1, df2], axis=1)

def morans_I_score():
    '''
    This function uses the Moran's I score to determine if an isotopolouge has a strong enough signal to be considered for prediction. 
    In statistics, Moran's I is a measure of spatial autocorrelation. Spatial autocorrelation is characterized by a correlation in a signal among
    nearby locations in space. 

    '''
    return 0

def count_list_elements(list_of_lists, cutoff = 3):
    '''
    This function was written to determine which isotopolouges are "relevant" across multiple replicates. It counts how many times each iso index appears and decides whether
    we should keep it or not.

    Parameters:
        - list_of_lists: a list of lists, where each sublist contains the indices for a single replicate brain (the elements are the indices of the isotopolouges)
    
    '''

    # Flatten the list of lists into single lists
    flat_list = [item for sublist in list_of_lists for item in sublist]
    flat_list.sort()

    # Returns a dictionary where the keys are the iso indices and the values are the number of times they appear in the flattened list (a count)
    iso_index_dict = Counter(flat_list)
    print(iso_index_dict)
    # List of the isos that appear an acceptable amount of times (deemed by cutoff)
    valid_isotopolouges = [index for index in list(iso_index_dict.keys()) if iso_index_dict[index] >= cutoff]
    valid_isotopolouges.sort()

    return valid_isotopolouges

def predict_and_plot(sample = "KD-M1", valid_isos = [], checkpoint_path = './saved-weights/train5-test1-m0-ranked/checkpoint'):
    ions_coord = get_data(file_name=f"/brain-m0-no-log/BG-{sample}-ioncounts-ranks.csv", keep_coord=True)
    isotopolouges_coord = get_data(file_name=f"/brain-m0-no-log/BG-{sample}-isotopolouges-ranks.csv", keep_coord=True)

    ions = get_data(file_name=f"/brain-m0-no-log/BG-{sample}-ioncounts-ranks.csv")
    isotopolouges = get_data(file_name=f"/brain-m0-no-log/BG-{sample}-isotopolouges-ranks.csv")

    isotopolouges_filtered = isotopolouges.iloc[:, valid_isos]
    predicted = predict(ions, isotopolouges_filtered, checkpoint_path=checkpoint_path)

    #predicted.index = isotopolouges.index
    isotopolouges_filtered[['x', 'y']] = isotopolouges_coord[['x', 'y']]
    predicted[['x', 'y']] = isotopolouges_coord[['x', 'y']]

    plot_multiple_brains(isotopolouges_filtered)
    plot_multiple_brains(predicted)

# ============================================== Train Uncertainty  ===========================================================
def train_uncertainty(training_features, training_targets, testing_features, testing_targets, ITERATIONS = 10, EPOCHS = 100, BATCH_SIZE = 128, file_name = "10"):
    '''
    Train the model 100 times and show results for top X percentile with probability >= Y% to measure uncertainty. This will be done by:
        - Training a model for EPOCHS amount of epochs
        -

    '''
    # Initialize np array to hold the predictions. Array should be 
    predictions = np.zeros((ITERATIONS, testing_targets.shape[0], testing_targets.shape[1]))

    # Learning rate scheduler 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose = 1, min_lr=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)

    for i in range(ITERATIONS):
        print(f'We are in iteration {i}')
        # if i % 10 == 0: print(i)
        num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val = preprocess_data(training_features, training_targets, testing_split = False)
        isotopolouge_names = list(training_targets.columns)

        # define model
        model = FML_regression_model(num_ion_counts, num_isotopolouges, 0.01)
        history = model.fit(x_train, y_train, batch_size = BATCH_SIZE, verbose=1, validation_data = (x_val, y_val), epochs=EPOCHS, callbacks=[reduce_lr, early_stopping])
        x_test = testing_features.to_numpy()
        prediction = model.predict(x_test)
        predictions[i] = prediction

   # np.save(f'uncertainty/{file_name}-data.npy', predictions)
    np.savez_compressed(f'uncertainty/{file_name}-data.npz', predictions)

    return predictions, isotopolouge_names

def print_evaluation_metrics(ground_truth_df, predicted_df, num_rows = 200, create_df = False, print_python_table = False, latex_table = False):
    '''
    Function to report regressional evaluation metrics for deep learning model. 
    '''   
    spearman_sorted = spearman_rankings(ground_truth_df, predicted_df, plot = False)
    top_predicted_metab_names = list(spearman_sorted['isotopologue'])
    median_rhos = list(spearman_sorted['median_rho'])
    pvals = list(spearman_sorted['pvals'])

    metric_names = ['isotopologue', 'Median Rho', 'p-value', 'MSE', 'MAE', 'R2']
    evaluation_metrics = []
    metabolites_used = set()
    mse_list = []
    mae_list = []
    r2_list = []

    for i, isotopologue_name in enumerate(top_predicted_metab_names): #[0:num_rows]):
        metabolite_name = isotopologue_name[0:-5] if not create_df else isotopologue_name
        if metabolite_name not in metabolites_used:
            ground_truth = ground_truth_df.loc[:, [isotopologue_name]]
            predicted = predicted_df.loc[:, [isotopologue_name]]

            mse = round(mean_squared_error(ground_truth, predicted), 4)
            mse_list.append(mse)

            mae = round(mean_absolute_error(ground_truth, predicted), 4)
            mae_list.append(mae)

            r_2 = round(r2_score(ground_truth, predicted), 4)
            r2_list.append(r_2)

            evaluation_metrics.append([isotopologue_name, round(median_rhos[i], 4), round(pvals[i], 4), mse, mae, r_2])

            if latex_table:
                print(f'{isotopologue_name} & {round(median_rhos[i], 4)} & {round(pvals[i], 4)} & {mse} & {mae} & {r_2} \\\\')
            
            metabolites_used.add(metabolite_name)

    if print_python_table:
        myTable = PrettyTable(metric_names)
        for i in range(len(evaluation_metrics)):
            myTable.add_row(evaluation_metrics[i])

        print(myTable)

    #spearman_sorted['MSE'] = mse_list
    #spearman_sorted['MAE'] = mae_list
    #spearman_sorted['R2'] = r2_list 

    return pd.DataFrame(evaluation_metrics, columns = metric_names)

def relative_metabolite_success(TIC_metabolite_names = 0, morans_invalid_isos = 0, isotopologue_metrics= 0, all_isotopologues = 0, num_bars = 65):
    '''
    parameters:
        - TIC_metabolite_names: a list of the metabolite names in the total ion counts matrix. The full metabolite list has 353. 
        - morans_invalid_isos: list of names of isotopologues that were identified by moran's i to be removed. 
        - isotopologue_metrics: df containing the name of each isotopologue and its regression evaluation metrics
    '''
    metabs_success_count = dict()
    metabs_set = set()
    isotopologue_metrics.sort_values(by=["isotopologue"], ascending=False, inplace = True)
    successful_metabs = []

    for index, row in isotopologue_metrics.iterrows():
        metab_name = row['isotopologue'][0:-5]
        
        if metab_name not in metabs_set:
            metabs_set.add(metab_name)
            metabs_success_count[metab_name] = [0,0,0]

        if row['R2'] >= 0.3:
            metabs_success_count[metab_name][0] += 1
        else:
            metabs_success_count[metab_name][1] += 1

    for isotopologue in all_isotopologues:
        metab_name = isotopologue[0:-5]
        if metab_name not in metabs_set:
            metabs_set.add(metab_name)
            metabs_success_count[metab_name] = [0,0,0]
        else:
            metabs_success_count[metab_name][2] += 1

    # print(metabs_success_count)
    stacked_bar_plot(metabs_success_count, num_bars=num_bars)#len(metabs_set))

    return isotopologue_metrics

# ======================================================== CROSS VALIDATION  ========================================================
def cross_validation_testing(all_invalid_isos, data_path = '/brain-m0-no-log', FML = True, tracer = 'BG', checkpoints_dir_label = "glucose", checkpoints_path = "./saved-weights", cutoff = 4):
    checkpoints_dir = f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}'
    # List of isotopologue names that should be removed from the dataset based on Moran's I score being low across majority of replicates
    invalid_morans = indices_to_metab_names(all_invalid_isos, cutoff = cutoff)
    # Lists of feature dataframes and target dataframes respectively, where each element is a different replicate
    ion_data, iso_data = remove_data_inconsistencies(additional_metabs_to_remove = invalid_morans, data_path = data_path, FML = FML, tracer = tracer)
    
    ground_truth = []
    predictions = []
    sorted_iso_names = []
    
    for i in range(len(ion_data)):
        # Create training and testing sets - pull one replicate out of the set 
        _, _, testing_features, testing_targets = create_full_dataset(ion_data, iso_data, holdout_index = i)
        print(f"Testing with replicate {i} heldout. # samples = {testing_features.shape[0]}")
        checkpoint_path = checkpoints_dir + f'/holdout-{i}/checkpoint'
        predicted = predict(testing_features, testing_targets, checkpoint_path = checkpoint_path)
        ground_truth.append(testing_targets)
        predictions.append(predicted)

        ion_data.insert(i, testing_features)
        iso_data.insert(i, testing_targets)

        # sorted_iso_names.append(list(spearman_rankings(testing_targets, predicted, plot = False)['isotopologue']))
        # plot_individual_isotopolouges_2(testing_targets, predicted, list(testing_targets.columns), specific_to_plot = sorted_iso_names[i][0:25], grid_size = 5, ranked = True)
   
    return ground_truth, predictions

def cross_validation_training(all_invalid_isos, data_path = '/brain-m0-no-log', FML = True, tracer = 'BG', checkpoints_dir_label = "glucose", checkpoints_path = "./saved-weights", cutoff = 4, EPOCHS = 100):
    '''
    Performs cross validation for isotopologue prediction model. Trains the model as many times as there are replicates in the dataset, holding out a different 
    replicate for testing each time. The saved weights for each model checkpoint is saved in the directory 

    Parameters:
        - all_invalid_isos (list): list of list, where each sublist contains the indices of isotopologues that did not pass the Moran's I test for a given replicate
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose
        - checkpoints_dir_label (string): string to append to the end of checkpoints directory name (dir name will be 'cross-validation-{checkpoints_dir_label}')
        - checkpoints_path (string): path to directory in which to house the checkpoint directory
    '''

    # List of isotopologue names that should be removed from the dataset based on Moran's I score being low across majority of replicates
    invalid_morans = indices_to_metab_names(all_invalid_isos, cutoff = cutoff, data_path=data_path, tracer=tracer)
    # Lists of feature dataframes and target dataframes respectively, where each element is a different replicate
    ion_data, iso_data = remove_data_inconsistencies(additional_metabs_to_remove = invalid_morans, data_path = data_path, FML = FML, tracer = tracer)
    # Create a directory wherever you are saving weights to hold subdirectories for each replicates checkpoints
    os.mkdir(f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}')

    for i in range(len(ion_data)):
        # Create training and testing sets - pull one replicate out of the set 
        training_features, training_targets, testing_features, testing_targets = create_full_dataset(ion_data, iso_data, holdout_index = i)
        print(f"Training with replicate {i} heldout. # samples = {training_features.shape[0]}")
        # Train the model 
        checkpoint = f'holdout-{i}'
        os.mkdir(f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}/{checkpoint}')
        history = training(training_features, training_targets, f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}/{checkpoint}/checkpoint', train = True, TRAIN_ENTIRE_BRAIN = True, EPOCHS = EPOCHS, BATCH_SIZE = 128)   
        
        # Reinsert the holdout replicate back into the fold to repeat the process with a different holdout. 
        ion_data.insert(i, testing_features)
        iso_data.insert(i, testing_targets)

        training_loss = history.history['loss']
        test_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history
        plt.figure(figsize=(3,3))
        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.title(f'Holdout: replicate {i}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.show();

    return 0 

def cross_validation_eval_metrics(ground_replicates, predicted_replicates, num_bars = 65):
    '''
    Takes the ground truth and predicted values of a series of replicates, concatenate them all and calculate the evaluation metrics as a form of cross validation. 

    parameters:
        - ground_replicates (list): list of ground truth isotopologue values (df of size # pixels x # isotopologues) where each element is a different replicate.
        - predicted_replicates (list): list of predicted isotopologue values (df of size # pixels x # isotopologues) where each element is a different replicate corresponding to the respective element in ground_replicates.
        - sorting_metric (string, default = "isotopologue"): name of the column that contains the isotopologue names.
        - eval_metric (string, default = "Median Rho"): name of the column with the sorting metric that the returned dataframe should be sorted by. 

    returns:
        - new_df (dataframe): df with columns [isotopologue, eval_metrics...] that contains the evaluation metrics taken for all replicates results concatenated together.
    '''

    num_replicates = len(ground_replicates)

    grounds = ground_replicates[0]
    for i, data in enumerate(ground_replicates[1:num_replicates]):
        grounds = pd.concat([grounds, data], ignore_index=True, axis = 0)

    predicts = predicted_replicates[0]
    for i, data in enumerate(predicted_replicates[1:num_replicates]):
        predicts = pd.concat([predicts, data], ignore_index=True, axis = 0)

    print(grounds.shape, predicts.shape)
    val_sorted_dataframe = spearman_rankings(grounds, predicts, plot=True)['isotopologue'] 

    df = print_evaluation_metrics(grounds, predicts, num_rows=200, create_df=True, latex_table=False)
    temp = relative_metabolite_success(isotopologue_metrics = df, all_isotopologues=list(grounds.columns), num_bars=num_bars)

    return val_sorted_dataframe

def cross_validation_average_metrics(ground_replicates, predicted_replicates, sorting_metric = "isotopologue", eval_metric = "Median Rho"):
    '''
    Takes the ground truth and predicted values of a series of replicates and calculated the average of the evaluation metrics for each replicate individually as a 
    form of cross validation. 

    parameters:
        - ground_replicates (list): list of ground truth isotopologue values (df of size # pixels x # isotopologues) where each element is a different replicate.
        - predicted_replicates (list): list of predicted isotopologue values (df of size # pixels x # isotopologues) where each element is a different replicate corresponding to the respective element in ground_replicates.
        - sorting_metric (string, default = "isotopologue"): name of the column that contains the isotopologue names.
        - eval_metric (string, default = "Median Rho"): name of the column with the sorting metric that the returned dataframe should be sorted by. 

    returns:
        - new_df (dataframe): df with columns [isotopologue, eval_metrics...] that contains the average of the evaluation metrics for each replicate.
    '''
    grounds_list = []
    for i in range(len(ground_replicates)):
        df_Val = print_evaluation_metrics(ground_replicates[i], predicted_replicates[i], num_rows=200, create_df=True).sort_values(by=[sorting_metric], ascending=False, ignore_index = True)       
        # Move the sorting_metric column to the front to be able to use iloc right after. 
        first_column = df_Val.pop(sorting_metric)
        df_Val.insert(0, sorting_metric, first_column)

        arr_1 = np.array(df_Val.iloc[:, 1:])
        grounds_list.append(arr_1)
        
        if i == 0:
            iso_names = df_Val.loc[:, [sorting_metric]]
            eval_headers = list(df_Val.columns)[1:]

    # print(iso_names)
    new_df = pd.DataFrame(np.mean(grounds_list, axis = 0), columns = eval_headers)
    new_df[sorting_metric] = iso_names
    new_df = new_df.sort_values(by=[eval_metric], ascending=False, ignore_index = True)

    return new_df

def identify_reproducibly_well_predicted(eval_list = []):
    well_predicted_individually = []

    for i, df in enumerate(eval_list):
        well_predicted_names = []

        for index, row in df.iterrows():
            if row['R2'] >= 0.3:
                isotopologue_name = row['isotopologue']
                well_predicted_names.append(isotopologue_name)

        well_predicted_individually.append(well_predicted_names)

    # Flatten the list of lists into single lists
    flat_list = [item for sublist in well_predicted_individually for item in sublist]
    flat_list.sort()

    # Returns a dictionary where the keys are the iso indices and the values are the number of times they appear in the flattened list (a count)
    iso_index_dict = Counter(flat_list)
    # List of the isos that appear an acceptable amount of times (deemed by cutoff)
    reproducible_isotopolouges = [index for index in list(iso_index_dict.keys()) if iso_index_dict[index] == len(eval_list)]
    reproducible_isotopolouges.sort()

    return reproducible_isotopolouges

# ======================================================== REWORKING FOR KIDNEYS  ========================================================

def map_poor_unlabeled_metabolites(metab_names, moransi_scores, replicate_cutoff = 3, morans_cutoff = 0.75):
    '''
    Takes in a list of lists with mappings of metabolite names to Moran's scores and returns which ones should be removed based on how many replicates fail the cutoff for that metabolite.
    All of the isotopologues for these metabolites will then be removed from consideration when training the model. 
    '''
    vaild_metab_names = []

    for replicate_number, replicate in enumerate(zip(metab_names, moransi_scores)):
        print(f"Working on replicate {replicate_number}")
        valid_metabs_replicate = [replicate[0][i] for i in range(len(replicate[0])) if replicate[1][i] >= morans_cutoff]
        vaild_metab_names.append(valid_metabs_replicate)

    # Flatten the list of lists into single lists - rework into separate function later
    flat_list = [item for sublist in vaild_metab_names for item in sublist]
    flat_list.sort()

    # Returns a dictionary where the keys are the iso indices and the values are the number of times they appear in the flattened list (a count)
    valid_name_dict = dict()
    for metab_name in flat_list:
        if metab_name in valid_name_dict:
            valid_name_dict[metab_name] += 1
        else:
            valid_name_dict[metab_name] = 1

    # List of the isos that appear an acceptable amount of times (deemed by cutoff)
    final_valid_metab_names = [index for index in list(valid_name_dict.keys()) if valid_name_dict[index] >= replicate_cutoff]
    final_valid_metab_names.sort()

    return final_valid_metab_names

def generate_invalid_iso_list(iso_paths, valid_metabolite_names):
    '''
    For each replicate, identify the names of the isotopologues that should be removed. Concat into a big list removing duplicates. 
    '''
    invalid_iso_names = []
    for i in range(len(iso_paths)):
        iso_data = get_data(iso_paths[i])
        iso_names = list(iso_data.keys())
        invalid_iso_names_replicate = [iso for iso in iso_names if iso[0:-5] not in valid_metabolite_names]

        invalid_iso_names.append(invalid_iso_names_replicate)

    # Flatten the list of lists into single lists - rework into separate function later
    flattened_invalid_iso_names = [item for sublist in invalid_iso_names for item in sublist]
    unique_invalid_iso_names = [*set(flattened_invalid_iso_names)]

    return unique_invalid_iso_names

def cross_validation_training_kidney(all_valid_metab_names, data_path = '/kidney-m0-no-log', FML = True, tracer = 'glutamine', checkpoints_dir_label = "cross-validation-KGlutamine", checkpoints_path = "./saved-weights", cutoff = 3, EPOCHS = 100):
    '''
    Performs cross validation for isotopologue prediction model. Trains the model as many times as there are replicates in the dataset, holding out a different 
    replicate for testing each time. The saved weights for each model checkpoint is saved in the directory 

    Parameters:
        - all_invalid_isos (list): list of list, where each sublist contains the indices of isotopologues that did not pass the Moran's I test for a given replicate
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose
        - checkpoints_dir_label (string): string to append to the end of checkpoints directory name (dir name will be 'cross-validation-{checkpoints_dir_label}')
        - checkpoints_path (string): path to directory in which to house the checkpoint directory
    '''

    ion_paths, iso_paths = generate_filepath_list(data_path = data_path, FML = FML, tracer = tracer)
    # List of isotopologue names that should be removed from the dataset based on Moran's I score being low across majority of replicates
    invalid_morans = generate_invalid_iso_list(iso_paths, all_valid_metab_names)
    
    # Lists of feature dataframes and target dataframes respectively, where each element is a different replicate
    ion_data, iso_data = remove_data_inconsistencies(additional_metabs_to_remove = invalid_morans, data_path = data_path, FML = FML, tracer = tracer)
    # Create a directory wherever you are saving weights to hold subdirectories for each replicates checkpoints
    os.mkdir(f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}')

    for i in range(len(ion_data)):
        # Create training and testing sets - pull one replicate out of the set 
        training_features, training_targets, testing_features, testing_targets = create_full_dataset(ion_data, iso_data, holdout_index = i)
        print(f"Training with replicate {i} heldout. # samples = {training_features.shape[0]}")
        # Train the model 
        checkpoint = f'holdout-{i}'
        os.mkdir(f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}/{checkpoint}')
        history = training(training_features, training_targets, f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}/{checkpoint}/checkpoint', train = True, TRAIN_ENTIRE_BRAIN = True, EPOCHS = EPOCHS, BATCH_SIZE = 128)   
        
        # Reinsert the holdout replicate back into the fold to repeat the process with a different holdout. 
        ion_data.insert(i, testing_features)
        iso_data.insert(i, testing_targets)

        training_loss = history.history['loss']
        test_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history
        plt.figure(figsize=(3,3))
        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.title(f'Holdout: replicate {i}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.show();

    return 0 

def cross_validation_testing_kidney(all_valid_metab_names, data_path = '/kidney-m0-no-log', FML = True, tracer = 'glutamine', checkpoints_dir_label = "cross-validation-KGlutamine", checkpoints_path = "./saved-weights", cutoff = 3):
    checkpoints_dir = f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}'
    ion_paths, iso_paths = generate_filepath_list(data_path = data_path, FML = FML, tracer = tracer)
    # List of isotopologue names that should be removed from the dataset based on Moran's I score being low across majority of replicates
    invalid_morans = generate_invalid_iso_list(iso_paths, all_valid_metab_names)
    # Lists of feature dataframes and target dataframes respectively, where each element is a different replicate
    ion_data, iso_data = remove_data_inconsistencies(additional_metabs_to_remove = invalid_morans, data_path = data_path, FML = FML, tracer = tracer)
    
    ground_truth = []
    predictions = []
    sorted_iso_names = []
    
    for i in range(len(ion_data)):
        # Create training and testing sets - pull one replicate out of the set 
        _, _, testing_features, testing_targets = create_full_dataset(ion_data, iso_data, holdout_index = i)
        print(f"Testing with replicate {i} heldout. # samples = {testing_features.shape[0]}")
        checkpoint_path = checkpoints_dir + f'/holdout-{i}/checkpoint'
        predicted = predict(testing_features, testing_targets, checkpoint_path = checkpoint_path)
        ground_truth.append(testing_targets)
        predictions.append(predicted)

        ion_data.insert(i, testing_features)
        iso_data.insert(i, testing_targets)

        # sorted_iso_names.append(list(spearman_rankings(testing_targets, predicted, plot = False)['isotopologue']))
        # plot_individual_isotopolouges_2(testing_targets, predicted, list(testing_targets.columns), specific_to_plot = sorted_iso_names[i][0:25], grid_size = 5, ranked = True)
   
    return ground_truth, predictions


# ======================================================== CROSS TISSUE PREDICTIONS ========================================================

def generate_valid_metabs(morans_path = 'valid-metabs-brain.txt', data_path = '/brain-m0-no-log/Brain-3HB', FML = True, tracer = 'B3HB', num_replicates = 6, morans_cutoff = 0.75): 
    '''
    For a given tracer/tissue combination, read in the metabolites and corresponding morans scores of each replicate. Use these to create a list of metabolites 
    that should be kept for model consideration based on the morans cutoff, and identify the isotopologues that need to be removed.
    
    Parameters:
        - morans_path (string): relative path from current directory to text file containing morans information (replicate metabolite names and morans scores) to be read in
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose
        - num_replicates (int): The number of replicates for this tracer -> how many replicates should match for a metabolite to be kept
        - morans_cutoff (float): The moran's I cutoff score for a metabolite to be kept. 

    Returns:
        - good_metabs
    '''

    # Load in the moran's I information from txt file
    f = open(morans_path, "r")
    lines = f.readlines()
    f.close()

    # Generate filepath list
    ion_paths, iso_paths = generate_filepath_list(data_path = data_path, FML = FML, tracer = tracer)

    # For each replicate, read in it's list of metabolites and their moran's scores and append to master list for processing. 
    metab_names = []
    moransi_scores = []

    # Since they are being read in from a txt, need to do some string processing to convert to proper list format.
    chars_to_remove = ["[", "'", "]"]

    for ion_path in ion_paths:
        # Convert the path name to match how it appears in the txt file
        ion_path = f'{ion_path[1:-20]}\n'
        # Obtain the filename index, use as reference point to access the metab names and morans scores. 
        index = lines.index(ion_path)

        # Read in the metabs and morans strings without the new line character
        metabs_string = lines[index+2][0:-1]    
        morans_string = lines[index+4][0:-1]

        # Remove the unnecessary characters so the string can be split. 
        # Extract the elements within the square brackets
        elements = metabs_string[metabs_string.index('[') + 1: metabs_string.index(']')]
        metabs_names = [element[1:-1] for element in elements.split(", ")]

        for char_to_remove in chars_to_remove:
            morans_string = morans_string.replace(char_to_remove, "")

        metab_names.append(metabs_names[0:-2])
        # Convert the morans scores from strings to floats
        moransi_scores.append(list(map(float, morans_string.split(", "))))


    good_metabs = map_poor_unlabeled_metabolites(metab_names, moransi_scores, replicate_cutoff = num_replicates, morans_cutoff = morans_cutoff)
    good_metabs.sort()

    # List of isotopologue names that should be removed from the dataset based on Moran's I score being low across majority of replicates
    '''
    invalid_morans = generate_invalid_iso_list(iso_paths, good_metabs)
    invalid_morans.sort()

    return good_metabs, invalid_morans
    '''
    return good_metabs

def preserve_good_metabs(good_metabolite_names, good_iso_names = [], data_path = '/brain-m0-no-log/Brain-3HB', FML = True, tracer = 'B3HB'):
    '''
    Generates the final dataset to use for regression. Loads in the relevant input files, and then removes two different sets of metabolites from each replicate:
        1). Metabolites/isotopologues that are not consistent across replicates (were not detected through mass spec for some replicates)
        2). Metabolites/isotopologues that were deemed invalid by failing to surpass the Moran's I metric for the majority of replicates

    Paremeters:
        - good_metabolite_names (list): list of metabolite NAMES (not indices) that will be kept during the regression. Any metabolites not listed, or isotopologues not
                                        belonging to those listed will be removed.
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose

    Returns:
        - clean_ion_data (list): list of ion count dataframes that are n (number of pixels in this replicate - can be different for each) by m (num of metabolites - consistent across all)
        - clean_iso_data (list): list of isotopologue dataframes that are n (number of pixels in this replicate - can be different for each) by m (num of isotopologues for prediction - consistent across all)
    '''

    # Lists of filepaths to ion_counts and isotopolouges
    ion_counts_paths, isotopolouges_paths = generate_filepath_list(data_path = data_path, FML = FML, tracer = tracer)
    # Lists of names of inconsistent  metabolites and isotopolouges that need to be removed
    ion_inconsistencies, iso_inconsistencies = checking_data_consistency(data_path = data_path, FML = FML, tracer = tracer)

    print(f"Inconsistencies found: {len(ion_inconsistencies)} metabolites, {len(iso_inconsistencies)} isotopolouges")

    clean_ion_data = []
    clean_iso_data = []

    # Load in the ion_count data
    for i, ion_count_path in enumerate(ion_counts_paths):
        # Load in the data for single replicate
        ion_data = get_data(file_name = ion_count_path, dir = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data')
        # Get list of metabolites for that replicate
        metabolite_names = list(ion_data.columns)
        # List of metabolites that must be dropped that are present in this replicate 
        metab_to_drop = [metab for metab in metabolite_names if metab in ion_inconsistencies or not metab in good_metabolite_names]
        # Drop the unneeded metabolites
        ion_data = ion_data.drop(labels = metab_to_drop, axis = 1)
        new_metabolite_names = ion_data.columns

        print(f"File {i}: {ion_count_path} || {len(metab_to_drop)} to drop || {len(metabolite_names) - len(new_metabolite_names)} dropped")

        # Append to list of cleaned/filtered dataframes for iso data

        clean_ion_data.append(ion_data)

    # Confirm that all the ion_count dataframes have the same columns (metabolites) in the same order!
    checks = [True if (list(item.columns) == list(clean_ion_data[0].columns)) else False for item in clean_ion_data[1:]]
    if all(checks):
        print("Ion-Data is all consistent! Time to train a model!")
    else:
        print("THERE HAS BEEN AN ERROR!!!! Dataframes columns not all the same order.")

    # For isotopologues, remove inconsistencies as well as any isotopolgoues that do not belong to the prime metabolites
    for i, iso_path in enumerate(isotopolouges_paths):
        iso_data = get_data(file_name = iso_path, dir = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data')
        iso_names = iso_data.columns
        if good_iso_names:
            iso_to_drop = [iso for iso in iso_names if iso in iso_inconsistencies or not iso[0:-5] in good_metabolite_names or not iso in good_iso_names]
        else:
            iso_to_drop = [iso for iso in iso_names if iso in iso_inconsistencies or not iso[0:-5] in good_metabolite_names]

        iso_data = iso_data.drop(labels = iso_to_drop, axis = 1)
        new_iso_names = iso_data.columns

        print(f"File {i}: {iso_path} || {len(iso_to_drop)} to drop || {len(iso_names) - len(new_iso_names)} dropped")

        clean_iso_data.append(iso_data)

    # Confirm that all the dataframes have the same columns in the same order!
    checks = [True if (list(item.columns) == list(clean_iso_data[0].columns)) else False for item in clean_iso_data[1:]]
    if all(checks):
        print("Iso-Data is all consistent! Time to train a model!")
    else:
        print("THERE HAS BEEN AN ERROR!!!! Dataframes columns not all the same order.")

    print("Reading in coord data")
    coords_df = [get_data(file_name=f"{path}", keep_coord=True).loc[:, ['x', 'y']] for path in ion_counts_paths]

    return clean_ion_data, clean_iso_data, new_metabolite_names, new_iso_names, coords_df

def train_full_tracer(ion_data, iso_data, checkpoints_dir_label = "3HB-cross-tissue", checkpoints_path = "./saved-weights", EPOCHS = 200 ):
    '''
    
    '''
    training_features, training_targets = create_full_dataset(ion_data, iso_data, holdout=False)
    history = training(training_features, training_targets, f'{checkpoints_path}/{checkpoints_dir_label}/checkpoint', train = True, TRAIN_ENTIRE_BRAIN = True, EPOCHS = EPOCHS, BATCH_SIZE = 128)   
    return history


# ======================================================== FEATURE IMPORTANCE for BLACK BOX MODEL ========================================================

def train_for_feature_importance(tracer = 'B3HB', num_replicates = 6, morans_path = 'valid-metabs-brain.txt', data_path = '/brain-m0-no-log/Brain-3HB', FML = True, morans_cutoff = 0.75):
    '''
    This pipeline is designed to be run with The Holdout Randomization Test for Feature Selection in Black Box Models proposed by Tansey et al (2021). 
    The HRT works with any predictive model, and uses data splitting to produce a valid p-value for each feature. The researchers compared the HRT to
    another approach and found that the HRT had better performance in terms of power and controlling the error rate. They applied the HRT to two real-life 
    examples and showed how it outperformed heuristic methods in selecting important features for predictive models.

    For a given tracer/tissue: 
        - Load in the data and normalize the features/targets using Moran's I Metric and Consistency
        - Repeatedly train two separate models holding out 1 metabolite each time:
            - A model to predict the held out metabolite from all the other metabolites 
            - A model to predict all of the isotopologues from all metabolites except the hold out metabolite

    Parameters:
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose
        - num_replicates (int): The number of replicates for this tracer -> how many replicates should match for a metabolite to be kept
        - morans_path (string): relative path from current directory to text file containing morans information (replicate metabolite names and morans scores) to be read in
        - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - morans_cutoff (float): The moran's I cutoff score for a metabolite to be kept. 
    '''

    # List of metabolites that should be kept for model consideration based on the morans cutoff, and identify the isotopologues that need to be removed.
    good_metabs = generate_valid_metabs(morans_path = morans_path, data_path = data_path, FML = FML, tracer = tracer, num_replicates = num_replicates, morans_cutoff = morans_cutoff)
    # Returns list of cleaned replicate data
    clean_ion_data, clean_iso_data, new_metabolite_names, new_iso_names, coords_df = preserve_good_metabs(good_metabs, data_path = data_path, FML = FML, tracer = tracer)
    # Compile individual cleaned replicates into final dataset to use for regression
    training_features, training_targets = create_full_dataset(clean_ion_data, clean_iso_data, holdout = False)

    # Iterate through all metabolites, remove one at a time and do the following: 
    for metabolite_number, metabolite in enumerate(list(training_features.columns)):
        print(metabolite_number, metabolite)
        # Predict the heldout metabolite from all of the other metabolites
        heldout_metab = training_features.loc[:, metabolite]
        remaining_metabs = training_features.loc[:, training_features.columns != metabolite]
        holdout_MSE = predict_heldout_metab(remaining_metabs, heldout_metab)

        # Predict the isotopologues from all metabolites minus the holdout
        num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val = preprocess_data(remaining_metabs, training_targets, testing_split = False)
        model = FML_regression_model(num_ion_counts, num_isotopolouges, 0.01)
        history = model.fit(x_train, y_train, batch_size = BATCH_SIZE, verbose=1, validation_data = (x_val, y_val), epochs=EPOCHS, callbacks=[model_checkpoint_callback])


    return training_features, training_targets

def NN_model_heldout_metab_from_remaining(num_remaining_metabolites, lambda_val):
    '''
    An extremely simple NN to predict a heldout metabolites values from the remaining metabolites. Used for a feature analysis.
    '''
    model = Sequential([
        # Input Layer
        Dense(128, input_dim = num_remaining_metabolites, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),

        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        
        Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        
        # Removed relu to allow negative 
        Dense(1, kernel_initializer='he_uniform', kernel_regularizer=l2(lambda_val))
    ])

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 3e-05),
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=['mse', 'mae'])

    return model

def predict_heldout_metab(feature_data, target_data, EPOCHS = 30, BATCH_SIZE = 256):
    num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val = preprocess_data(feature_data, target_data, testing_split = False)

    # define model
    model = NN_model_heldout_metab_from_remaining(num_ion_counts, 0.01)

    # fit model
    history = model.fit(x_train, y_train, batch_size = BATCH_SIZE, verbose = 2, validation_data = (x_val, y_val), epochs=EPOCHS)
    
    prediction = model.predict(x_val)
    MSE = np.square(np.subtract(y_val,prediction)).mean()

    return MSE


def main():
    path = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-isotopolouges.csv'
    isotopolouges = pd.read_csv(path) 
    isotopolouges = isotopolouges.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    print(isotopolouges)
    isotopolouge_names = list(isotopolouges.columns)

    ion_counts = pd.read_csv('/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-ioncounts.csv')
    ion_counts = ion_counts.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    print(ion_counts)
    metabolite_names = list(ion_counts.columns)

    x = ion_counts.to_numpy()
    y = isotopolouges.to_numpy()
    print(x.shape, y.shape)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)
    num_ion_counts = x.shape[1]
    num_isotopolouges = y.shape[1]
    print(num_ion_counts, num_isotopolouges)
    print(x_train, y_train)


    # define model
    model = multiple_regression(num_ion_counts, num_isotopolouges, 0.01)

    # Checkpoints
    # https://keras.io/api/callbacks/model_checkpoint/
    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose = 1,
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_best_only=False,
        save_freq= int(217 * 10))

    # fit model
    # history = model.fit(x_train, y_train, batch_size = 64, verbose=1, validation_data = (x_val, y_val), epochs=100, callbacks=[model_checkpoint_callback])


    model.load_weights(checkpoint_filepath)
    # evaluate model on test set
    mae = model.evaluate(x_test, y_test, verbose=1)

    prediction = model.predict(x_test)
    #plt.scatter(y_test, prediction)
    # plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-predicting.png')
    #plt.show()

    print(y_test.shape, prediction.shape)

    plot_individual_isotopolouges(y_test, prediction, isotopolouge_names)


if __name__ == "__main__":
    #main()
    print("Hi")
    print(tf.__version__)
    train = False
    model = multiple_regression_model(5, 5, 4)

    if False:
        # Single brain, unranked - trained on KD M1. Withold some of the brain as test set.
        brain_K1_ions = get_data(file_name="testing-all-ion.csv")
        brain_K1_isotopolouges = get_data(file_name="testing-all-isotopolouges.csv")
        # training(brain_K1_ions, brain_K1_isotopolouges, './saved-weights/unnormalized/KD-M1-unranked/checkpoint', train = True, TRAIN_ENTIRE_BRAIN = True)
        test_whole_brain(brain_K1_ions, brain_K1_isotopolouges, './saved-weights/unnormalized/KD-M1-unranked/checkpoint')

    if False:
        # Single brain, unranked - trained on KD M1. Withold some of the brain as test set.
        brain_K1_ions = get_data(file_name="brain-glucose-KD-M1-ioncounts.csv")
        brain_K1_isotopolouges = get_data(file_name="brain-glucose-KD-M1-isotopolouges.csv")
        training(brain_K1_ions, brain_K1_isotopolouges, './saved-weights/KD-M1-unranked-dropout/checkpoint', train = train, TRAIN_ENTIRE_BRAIN = False)
        test_whole_brain(brain_K1_ions, brain_K1_isotopolouges, './saved-weights/KD-M1-unranked-dropout/checkpoint')

    if False:
        # Test KD-M1 on entire KD-M2 and entire KD-M1
        brain_K2_ions = get_data(file_name="brain-glucose-KD-M2-ioncounts.csv")
        brain_K2_isotopolouges = get_data(file_name="brain-glucose-KD-M2-isotopolouges.csv")
        test_whole_brain(brain_K2_ions, brain_K2_isotopolouges, './saved-weights/KD-M1-unranked-dropout/checkpoint')

    if False: 
        ions, isotopolouges, feat, targets = create_large_data(all_data = False)    
        print(ions, isotopolouges)
        training(ions, isotopolouges, './saved-weights/unranked-train5-test1/checkpoint', train = True)

    if False:
        # Test first 5 brains on entire ND-M3
        brain_N3_ions = get_data(file_name="brain-glucose-ND-M3-ioncounts.csv")
        brain_N3_isotopolouges = get_data(file_name="brain-glucose-ND-M3-isotopolouges.csv")
        test_whole_brain(brain_N3_ions, brain_N3_isotopolouges, './saved-weights/unranked-train5-test1/checkpoint')

    # Ranked 
    if False:
        # Single brain, unranked - trained on KD M1. Withold some of the brain as test set.
        brain_K1_ranked_ions = get_data(file_name="brain-glucose-KD-M1-ioncounts-ranks.csv")
        brain_K1_ranked_isotopolouges = get_data(file_name="brain-glucose-KD-M1-isotopolouges-ranks.csv")
        brain_K1_ranked_ions = brain_K1_ranked_ions * 100
        brain_K1_ranked_isotopolouges = brain_K1_ranked_isotopolouges * 100
        print(brain_K1_ranked_isotopolouges)
        training(brain_K1_ranked_ions, brain_K1_ranked_isotopolouges, './saved-weights/KD-M1-ranked-dropout-scaled/checkpoint', train = train, TRAIN_ENTIRE_BRAIN = False)
        test_whole_brain(brain_K1_ranked_ions, brain_K1_ranked_isotopolouges, './saved-weights/KD-M1-ranked-dropout-scaled/checkpoint')

    if False:
        # Single brain, unranked - trained on KD M1. Withold some of the brain as test set.
        brain_K2_ranked_ions = get_data(file_name="brain-glucose-KD-M2-ioncounts-ranks.csv")
        brain_K2_ranked_isotopolouges = get_data(file_name="brain-glucose-KD-M2-isotopolouges-ranks.csv")
        training(brain_K2_ranked_ions, brain_K2_ranked_isotopolouges, './saved-weights/KD-M2-ranked-dropout/checkpoint', train = False, TRAIN_ENTIRE_BRAIN = False)
        test_whole_brain(brain_K2_ranked_ions, brain_K2_ranked_isotopolouges, './saved-weights/KD-M2-ranked-dropout/checkpoint')

    if False: 
        ions, isotopolouges, feat, targets = create_large_data_ranked(all_data = False)    
        print(ions, isotopolouges)
        training(ions, isotopolouges, './saved-weights/ranked-train5-test1/checkpoint', train = True)

    if False:
        # Single brain, unranked - trained on KD M1. Withold some of the brain as test set.
        brain_M1_ranked_ions = get_data(file_name="brain-glucose-ND-M1-ioncounts-ranks.csv")
        brain_M1_ranked_isotopolouges = get_data(file_name="brain-glucose-ND-M1-isotopolouges-ranks.csv")
        brain_M1_ranked_ions = brain_M1_ranked_ions * 100
        brain_M1_ranked_isotopolouges = brain_M1_ranked_isotopolouges * 100
        print(brain_M1_ranked_isotopolouges)
        training(brain_M1_ranked_ions, brain_M1_ranked_isotopolouges, './saved-weights/ND-M1-ranked-dropout-scaled/checkpoint', train = False, TRAIN_ENTIRE_BRAIN = False)
        test_whole_brain(brain_M1_ranked_ions, brain_M1_ranked_isotopolouges, './saved-weights/ND-M1-ranked-dropout-scaled/checkpoint')

        # Have the model transform across each feature by column  