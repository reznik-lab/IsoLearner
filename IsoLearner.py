import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Add, Dense, Activation, BatchNormalization, Lambda, ReLU, PReLU, LayerNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display
import seaborn as sns
from scipy import stats
from visualization import *
from collections import Counter
from statsmodels.stats.multitest import multipletests
import copy
from skimage.metrics import structural_similarity as ssim

from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor # for building the model
from prettytable import PrettyTable
import os
import re

# ======================================================== ISOLEARNER ========================================================

class IsoLearner:
    def __init__(self, 
                absolute_data_path, 
                relative_data_path, 
                morans_path = 'valid-metabs-brain.txt', 
                tracer = 'B3HB', 
                FML = True, 
                num_replicates = 6,
                morans_strat = 1, 
                morans_cutoff = 0.6,
                min_replicates = 3, 
                shared_morans_file = False):
        
        '''
        Parameters:
        - absolute_data_path (str): Absolute path to the directory containing the file (exclude trailing forward slash)
        - relative_data_path (str): relative path from absolute data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - morans_path (str): relative path from current directory to text file containing morans information (replicate metabolite names and morans scores) to be read in
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - num_replicates (int): The number of replicates for this tracer -> how many replicates should match for a metabolite to be kept
        - morans_strat (int): indicator for which moran's I strategy to use (detailed in generate_valid_metabs)
        - morans_cutoff (float): The moran's I cutoff score for a metabolite to be kept. 
        '''

        self.absolute_data_path = absolute_data_path
        self.relative_data_path = relative_data_path
        self.morans_path = morans_path
        self.tracer = tracer
        self.FML = FML
        self.num_replicates = num_replicates
        self.morans_strat = morans_strat
        self.morans_cutoff = morans_cutoff
        self.min_replicates = min_replicates
        self.shared_morans_file = shared_morans_file

        print("Initializing IsoLearner")

        # Generate filepath list
        self.ion_paths, self.iso_paths = self.generate_filepath_list()

        if self.morans_strat == 1:
            # Generate list of metabolites to be kept for imputation
            print("Generating List of valid metabolites from Moran's I calculations ", end = "")
            self.valid_metabolites = self.generate_valid_metabs()
            
            # Returns list of cleaned replicate data
            print("Cleaning data ", end = "")
            self.clean_ion_data, self.clean_iso_data, self.new_metabolite_names, self.new_iso_names, self.coords_df = self.preserve_good_metabs()
            self.all_models = []
        else:
            print("Cleaning data with strategy 2: ", end = "")
            _ = self.load_morans_data()
            self.metab_dict, self.good_iso_list = self.morans_df_to_dict(morans_cutoff=0.6)
            self.clean_ion_data, self.clean_iso_data, self.new_metabolite_names, self.new_iso_names, self.coords_df = self.preserve_good_metabs(good_iso_names=self.good_iso_list)
        
    # ============================================== LIST OF FILEPATHS =====================================================================
    def generate_filepath_list(self):
        '''
        Returns relative paths of data files as two lists. If sample includes both normal and ketogenic replicates, the ND replicates are first, and then KD. 
            - File names are expected to be in the form '{Tissue (B for brain)} - {Diet type (KD or ND)} - {Replicate Number} - {FML (full metabolite list)} - {ion counts or isotopologues} - {ranks}.csv"
            - Example Filename: 'B3HB-KD-M1-FML-ioncounts-ranks.csv'

        Returns: 
            - ion_counts_paths (list): list of filenames with ion_count data
            - isotopologues_paths (list): list of filenames with iso data
        '''

        # Qualifier for whether the full metabolite list or partial metabolite list is being used
        iso_path = 'FML-isotopolouges-ranks' if self.FML else 'isotopolouges-ranks'
        ion_path = 'FML-ioncounts-ranks' if self.FML else 'ioncounts-ranks'

        # List containing the file names of the isotopolouge data - normal diet mice
        isotopologues_paths = [f'{self.absolute_data_path}/{self.relative_data_path}/{self.tracer}-ND-M{i+1}-{iso_path}.csv' for i in range(3)]
        # List containing the file names of the ion count data - normal diet mice
        ion_counts_paths = [f'{self.absolute_data_path}/{self.relative_data_path}/{self.tracer}-ND-M{i+1}-{ion_path}.csv' for i in range(3)]

        # These two tracers have Ketogenic mice as well, include them in the filepaths
        if self.tracer == 'BG' or self.tracer == 'B3HB':
            isotopologues_paths.extend([f'{self.absolute_data_path}/{self.relative_data_path}/{self.tracer}-KD-M{i+1}-{iso_path}.csv' for i in range(3)])
            ion_counts_paths.extend([f'{self.absolute_data_path}/{self.relative_data_path}/{self.tracer}-KD-M{i+1}-{ion_path}.csv' for i in range(3)])

        return ion_counts_paths, isotopologues_paths

    # ============================================== IMPORT DATA FROM CSV =====================================================================
    def get_data(self, file_name, keep_coord = False, full_path = False):
        '''
        Convert file from csv to dataframe and remove unnecessary columns 

        Parameters:
            - file_name: name of the file
            - dir: Absolute path to the directory containing the file (exclude trailing forward slash)
        
        Returns:
            - data: dataframe of the data
        '''

        if full_path:
            data_path = file_name
        else:
            data_path = f'{self.absolute_data_path}/{self.relative_data_path}/{file_name}'
        data = pd.read_csv(data_path)

        if keep_coord:
            data = data.drop(labels = ['Unnamed: 0'], axis = 1)
        else:
            data = data.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
        return data 


    # ============================================== IDENTIFYING ION + ISO INCONSISTENCIES ============================================================
    def checking_data_consistency(self):
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

        # List of ions that need to be removed from all files
        ion_inconsistencies = self.identify_inconsistencies(self.ion_paths, show_progress = False)
        # List of isotopolouges that need to be removed from all files
        iso_inconsistencies = self.identify_inconsistencies(self.iso_paths, show_progress = False)

        return ion_inconsistencies, iso_inconsistencies

    # ============================================== IDENTIFYING DATA INCONSISTENCIES ============================================================
    def identify_inconsistencies(self, list_of_paths, show_progress = True):
        '''
        Helper Function - Goes through multiple datafiles and identifies metabolites (columns) that are not consistent between all files. 
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
            ion_count = self.get_data(file_name = name, full_path = True)
            metab_names = ion_count.columns

            if show_progress:
                print(i, name, len(metab_names))

            individual_replicate_metabs.append(metab_names)
            all_metabs.extend(metab_names)

            print("==", end = '')

        # Flatten the list of lists into single lists
        all_metabs.sort()

        # Returns a dictionary where the keys are the iso indices and the values are the number of times they appear in the flattened list (a count)
        metab_index_dict = Counter(all_metabs)
        # Create a list the names of all metabolites that do not appear in all replicates
        invalid_metabs_names = [index for index in list(metab_index_dict.keys()) if metab_index_dict[index] < num_replicates]

        return invalid_metabs_names
    

    # ===================================================================================================================================
    # <============================================== VALID METABS FROM MORANS TXT =====================================================>
    def generate_valid_metabs(self): 
        '''
        For a given tracer/tissue combination, read in the metabolites and corresponding morans scores of each replicate. Use these to create a list of metabolites 
        that should be kept for model consideration based on the morans cutoff, and identify the isotopologues that need to be removed.
        
        There are two modes this function will operate in, depending on the Moran's I Strategy being implemented (morans_strat):
            1). morans_strat = (1) cull_bloodline: Using this strategy, Moran's I is taken for all of the METABOLITES (ion_counts)
                - Metabolites that fail the metric are removed from consideration, as well as all of their isotopologues (their "bloodline")
            2). morans_strat = (2) darwinism: In this strategy, Moran's I is taken on all of the ISOTOPOLOGUES (isos)
                - All of the metabolites are kept, and only isotopologues that fail the metric are discarded. 

        This function DOES NOT actually load in any data at all, it simply reads in from the txt file. 

        Possible point of failure: the way the first line of each replicate entry in the txt file is named could have an effect on it working properly. Put some jank for now. 

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
        f = open(self.morans_path, "r")
        lines = f.readlines()
        f.close()

        # For each replicate, read in it's list of metabolites and their moran's scores and append to master list for processing. 
        metab_names = []
        moransi_scores = []

        # Since they are being read in from a txt, need to do some string processing to convert to proper list format.
        chars_to_remove = ["[", "'", "]"]

        # Whether we should load in the ion_count data or isotopologue data based on moran's strat being implemented
        data_file_paths = self.ion_paths if self.morans_strat == 1 else self.iso_paths
        for data_path in data_file_paths:
            # Convert the path name to match how it appears in the txt file           
            data_path = data_path.replace(self.absolute_data_path,'')
            data_path = f"{'-'.join(data_path[1:].split('-')[:-2])}\n"

            # Obtain the filename index, use as reference point to access the metab names and morans scores. 
            index = lines.index(data_path)
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


        good_metabs = self.map_poor_unlabeled_metabolites(metab_names, moransi_scores)
        good_metabs.sort()

        return good_metabs

    def map_poor_unlabeled_metabolites(self, metab_names, moransi_scores):
        '''
        Takes in a list of lists with mappings of metabolite names to Moran's scores and returns which ones should be removed based on how many replicates fail the cutoff for that metabolite.
        All of the isotopologues for these metabolites will then be removed from consideration when training the model. 
        '''
        vaild_metab_names = []

        for replicate_number, replicate in enumerate(zip(metab_names, moransi_scores)):
            print("==", end = '')
            # print(f"Working on replicate {replicate_number}")
            valid_metabs_replicate = [replicate[0][i] for i in range(len(replicate[0])) if replicate[1][i] >= self.morans_cutoff]
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
        final_valid_metab_names = [index for index in list(valid_name_dict.keys()) if valid_name_dict[index] >= self.num_replicates]
        final_valid_metab_names.sort()

        print("> Valid Metabolites Calculated")
        return final_valid_metab_names

    # <============================================== VALID METABS FROM TXT ============================================================>
    # ===================================================================================================================================

    def preserve_good_metabs(self, good_iso_names = []):
        '''
        Generates the final dataset to use for regression. Loads in the relevant input files, and then removes two different sets of metabolites from each replicate:
            1). Metabolites/isotopologues that are not consistent across replicates (were not detected through mass spec for some replicates)
            2). Metabolites/isotopologues that were deemed invalid by failing to surpass the Moran's I metric for the majority of replicates

        Paremeters:
            - self.valid_metabolites (list): list of metabolite NAMES (not indices) that will be kept during the regression. Any metabolites not listed, or isotopologues not
                                            belonging to those listed will be removed.
            - data_path (string): relative path from main data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
            - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
            - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
                - Precuror 'B' stands for brain data, 'G' for Glucose

        Returns:
            - clean_ion_data (list): list of ion count dataframes that are n (number of pixels in this replicate - can be different for each) by m (num of metabolites - consistent across all)
            - clean_iso_data (list): list of isotopologue dataframes that are n (number of pixels in this replicate - can be different for each) by m (num of isotopologues for prediction - consistent across all)
        '''

        # Lists of names of inconsistent metabolites and isotopolouges that need to be removed
        ion_inconsistencies, iso_inconsistencies = self.checking_data_consistency()

        print("> The dataset has been checked for inconsistencies")
        print(f"Inconsistencies found: {len(ion_inconsistencies)} metabolites, {len(iso_inconsistencies)} isotopolouges")

        clean_ion_data = []
        clean_iso_data = []

        # Load in the ion_count data
        for i, ion_count_path in enumerate(self.ion_paths):
            # Load in the data for single replicate
            ion_data = self.get_data(file_name = ion_count_path, full_path = True)
            # Get list of metabolites for that replicate
            metabolite_names = list(ion_data.columns)

            # List of metabolites that must be dropped that are present in this replicate - drop based on Moran's strat
            if self.morans_strat == 1:
                metab_to_drop = [metab for metab in metabolite_names if metab in ion_inconsistencies or not metab in self.valid_metabolites]
            else:
                metab_to_drop = [metab for metab in metabolite_names if metab in ion_inconsistencies]

            # Drop the unneeded metabolites
            ion_data = ion_data.drop(labels = metab_to_drop, axis = 1)
            new_metabolite_names = ion_data.columns

            print(f"File {i}: {ion_count_path} || {len(metab_to_drop)} to drop || {len(metabolite_names) - len(new_metabolite_names)} dropped")

            # Append to list of cleaned/filtered dataframes for iso data

            clean_ion_data.append(ion_data)

        # Confirm that all the ion_count dataframes have the same columns (metabolites) in the same order!
        checks = [True if (list(item.columns) == list(clean_ion_data[0].columns)) else False for item in clean_ion_data[1:]]
        if all(checks):
            print("Ion-Data is all consistent! Time to train a model!", end = "\n\n")
        else:
            print("THERE HAS BEEN AN ERROR!!!! Dataframes columns not all the same order.")

        # For isotopologues, remove inconsistencies as well as any isotopolgoues that do not belong to the prime metabolites
        for i, iso_path in enumerate(self.iso_paths):
            iso_data = self.get_data(file_name = iso_path, full_path = True)
            iso_names = iso_data.columns

            if good_iso_names:
                iso_to_drop = [iso for iso in iso_names if iso in iso_inconsistencies or not iso in good_iso_names]
            elif self.morans_strat == 2:
                iso_to_drop = [iso for iso in iso_names if iso in iso_inconsistencies or iso not in self.good_isotopologues]
            else:
                iso_to_drop = [iso for iso in iso_names if iso in iso_inconsistencies or not iso[0:-5] in self.valid_metabolites]


            iso_data = iso_data.drop(labels = iso_to_drop, axis = 1)
            new_iso_names = iso_data.columns

            print(f"File {i}: {iso_path} || {len(iso_to_drop)} to drop || {len(iso_names) - len(new_iso_names)} dropped")

            clean_iso_data.append(iso_data)

        # Confirm that all the dataframes have the same columns in the same order!
        checks = [True if (list(item.columns) == list(clean_iso_data[0].columns)) else False for item in clean_iso_data[1:]]
        if all(checks):
            print("Iso-Data is all consistent! Time to train a model!", end = "\n\n")
        else:
            print("THERE HAS BEEN AN ERROR!!!! Dataframes columns not all the same order.")

        print("Reading in coord data")
        coords_df = [self.get_data(file_name=f"{path}", keep_coord=True, full_path = True).loc[:, ['x', 'y']] for path in self.ion_paths]

        return clean_ion_data, clean_iso_data, new_metabolite_names, new_iso_names, coords_df



    # ============================================== Creating Dataset for Training  ===========================================================
    def create_full_dataset(self, ion_dfs, iso_dfs, holdout = True, holdout_index = 0):
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
            test_ion_counts = self.clean_ion_data.pop(holdout_index)
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


    # ===================================================================================================================================
    # <==================================================== TRAINING ===================================================================>
    def FML_regression_model(self, num_ion_counts, num_isotopolouges, lambda_val):
        model = Sequential([
            # Original: 128, 128, 256, 256, 256, 256, 128
            # Layer 1
            Dense(256, input_dim = num_ion_counts, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
            BatchNormalization(),

            Dense(512, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
            BatchNormalization(),
            
            Dense(512, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(512, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
            BatchNormalization(),
            
            Dense(512, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
            BatchNormalization(),
            
            Dense(512, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
            BatchNormalization(),
            Dropout(0.25),      
            
            Dense(512, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
            BatchNormalization(),
            
            # Removed relu to allow negative 
            Dense(num_isotopolouges, kernel_initializer='he_uniform', kernel_regularizer=l2(lambda_val))
        ])

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 3e-05),
                    #loss=tf.keras.losses.MeanSquaredError(),
                    loss = tf.keras.losses.MeanSquaredError(),
                    metrics=['mse', 'mae'])

        return model

    def preprocess_data(self, ion_counts, isotopolouges, testing_split = True):
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

    def training(self, feature_data, target_data, checkpoint_path = './saved-weights/KD-M1-unranked-dropout/checkpoint', train = True, TRAIN_ENTIRE_BRAIN = False, ranked = True, EPOCHS = 100, BATCH_SIZE = 32):
        print("HI")
        # Whether or not to treat the dataset as training data
        if TRAIN_ENTIRE_BRAIN:
            num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val = self.preprocess_data(feature_data, target_data, testing_split = False)
        else:
            num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val, x_test, y_test = self.preprocess_data(feature_data, target_data)
        
        isotopolouge_names = list(target_data.columns)
        # print(isotopolouge_names)

        # define model
        model = self.FML_regression_model(num_ion_counts, num_isotopolouges, 0.01)

        # Checkpoints
        # https://keras.io/api/callbacks/model_checkpoint/
        checkpoint_filepath = checkpoint_path 
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

            print(y_test.shape, prediction.shape)
            plot_individual_isotopolouges_2(y_test, prediction, isotopolouge_names, ranked = ranked)

        return history
   
    def cross_validation_training(self, checkpoints_dir_label = "3HB", checkpoints_path = "./saved-weights", EPOCHS = 100):
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
        print("WE ARE IN THIS FUNCTION")

        # Create a directory wherever you are saving weights to hold subdirectories for each replicates checkpoints
        full_checkpoint_dir = f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}'
        if os.path.exists(full_checkpoint_dir):
            # Check if the directory is empty
            if not os.listdir(full_checkpoint_dir):
                # Directory is empty, delete it
                try:
                    os.rmdir(full_checkpoint_dir)
                    print(f"Empty directory '{full_checkpoint_dir}' has been deleted.")
                except OSError as e:
                    print(f"Error deleting directory '{full_checkpoint_dir}': {e}")
            else:
                print(f"Directory '{full_checkpoint_dir}' is not empty. Aborting training.")
                return 0

        print("Creating checkpoint directory")
        os.mkdir(full_checkpoint_dir)

        for i in range(len(self.clean_ion_data)):
            # Create training and testing sets - pull one replicate out of the set 
            training_features, training_targets, testing_features, testing_targets = self.create_full_dataset(self.clean_ion_data, self.clean_iso_data, holdout_index = i)
            print(f"Training with replicate {i} heldout. # samples = {training_features.shape[0]}")

            # Train the model 
            checkpoint = f'holdout-{i}'
            os.mkdir(f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}/{checkpoint}')
            history = self.training(training_features, training_targets, f'{full_checkpoint_dir}/{checkpoint}/checkpoint', train = True, TRAIN_ENTIRE_BRAIN = True, EPOCHS = EPOCHS, BATCH_SIZE = 128)

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

            # Reinsert the holdout replicate back into the fold to repeat the process with a different holdout. 
            self.clean_ion_data.insert(i, testing_features)
            self.clean_iso_data.insert(i, testing_targets)
        
        return 0 

    # <==================================================== TRAINING ===================================================================>
    # ===================================================================================================================================

    # ===================================================================================================================================
    # <==================================================== EVALUATION ===================================================================>
    def predict(self, feature_data, target_data, checkpoint_path = './saved-weights/KD-M1-unranked-dropout/checkpoint'):
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
        model = self.FML_regression_model(num_ion_counts, num_isotopolouges, 0.01)

        # Checkpoints
        # https://keras.io/api/callbacks/model_checkpoint/
        checkpoint_filepath = checkpoint_path # './saved-weights/KD-M1-unranked-dropout/checkpoint'
        model.load_weights(checkpoint_filepath).expect_partial()
    
        prediction = model.predict(features)
        prediction_df = pd.DataFrame(prediction, columns = isotopolouge_names)

        return prediction_df
    
    def spearman_rankings(self, actual, predicted, plot = True):
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
        # color = ["purple" if pval <= 0.1 else "red" for pval in corrected_pvals[1]]
        color = ["purple" if median_rho >= 0.6 else "red" for median_rho in spearmans]
        df_spearman["color"] = color

        # Pull p-value 
        if plot:
            median_rho_feature_plot(df_spearman)

        sorted = df_spearman.sort_values(by=["median_rho"], ascending=False)

        return sorted
    
    def print_evaluation_metrics(self, ground_truth_df, predicted_df, num_rows = 200, create_df = False, print_python_table = False, latex_table = False):
        '''
        Function to report regressional evaluation metrics for deep learning model. 
        '''   
        spearman_sorted = self.spearman_rankings(ground_truth_df, predicted_df, plot = False)
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

    def relative_metabolite_success(self, TIC_metabolite_names = 0, morans_invalid_isos = 0, isotopologue_metrics= 0, all_isotopologues = 0, num_bars = 51, plot = True):
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
        myKeys = list(metabs_success_count.keys())
        myKeys.sort()
        sorted_dict = {i: metabs_success_count[i] for i in myKeys}

        # Define a custom sorting key function
        def custom_sort_key(item):
            return (item[0], -item[1])

        # Sort the dictionary by applying the custom key function
        sorted_items = sorted(sorted_dict.items(), key=lambda x: custom_sort_key(x[1]))

        # Create a new dictionary from the sorted items
        sorted_dict = dict(sorted_items)
        if plot:
            stacked_bar_plot(sorted_dict, num_bars=num_bars, plot_total_success= False)#len(metabs_set))

        return isotopologue_metrics, metabs_success_count
    
    def cross_validation_testing(self, checkpoints_dir_label = '3HB', checkpoints_path = "./saved-weights"):
        checkpoints_dir = f'{checkpoints_path}/cross-validation-{checkpoints_dir_label}'
        # List of isotopologue names that should be removed from the dataset based on Moran's I score being low across majority of replicates
        #invalid_morans = indices_to_metab_names(all_invalid_isos, cutoff = cutoff)
        # Lists of feature dataframes and target dataframes respectively, where each element is a different replicate
        #ion_data, iso_data = remove_data_inconsistencies(additional_metabs_to_remove = invalid_morans, data_path = data_path, FML = FML, tracer = tracer)
    
        ground_truth = []
        predictions = []
        sorted_iso_names = []
    
        for i in range(len(self.clean_ion_data)):
            # Create training and testing sets - pull one replicate out of the set 
            _, _, testing_features, testing_targets = self.create_full_dataset(self.clean_ion_data, self.clean_iso_data, holdout_index = i)
            print(f"Testing with replicate {i} heldout. # samples = {testing_features.shape[0]}")
            checkpoint_path = checkpoints_dir + f'/holdout-{i}/checkpoint'
            predicted = self.predict(testing_features, testing_targets, checkpoint_path = checkpoint_path)
            ground_truth.append(testing_targets)
            predictions.append(predicted)

            self.clean_ion_data.insert(i, testing_features)
            self.clean_iso_data.insert(i, testing_targets)

            # sorted_iso_names.append(list(spearman_rankings(testing_targets, predicted, plot = False)['isotopologue']))
            # plot_individual_isotopolouges_2(testing_targets, predicted, list(testing_targets.columns), specific_to_plot = sorted_iso_names[i][0:25], grid_size = 5, ranked = True)
   
        return ground_truth, predictions
   
    def cross_validation_eval_metrics(self, ground_replicates, predicted_replicates, num_bars = 88, plot = True):
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
        _ = self.spearman_rankings(grounds, predicts, plot=plot)['isotopologue'] 
        print("Done with spearmans")
        df = self.print_evaluation_metrics(grounds, predicts, num_rows=200, create_df=True, latex_table=False)
        metab_dict = self.relative_metabolite_success(isotopologue_metrics = df, all_isotopologues=list(grounds.columns), num_bars=num_bars, plot = plot)

        return metab_dict
        # return val_sorted_dataframe
        # return df 

    # <==================================================== EVALUATION ===================================================================>
    # ===================================================================================================================================

    # =================================================================================================================================================
    # <==================================================== BLACK BOX FEATURE EVAL ===================================================================>
    def black_box_feature_evaluation(self, metabolites = []):
        '''
        This function will use a set amount of metabolites and their corresponding isotopologues to test the black box feature evaluation model
        '''
        full_metabolites, full_isotopologues = self.create_full_dataset(self.clean_ion_data, self.clean_iso_data, holdout=False)
        shortened_metabolites = full_metabolites[metabolites]

        full_isotopologues_names = full_isotopologues.columns.to_list()
        print(metabolites)
        print(full_isotopologues_names[0])

        print([iso for iso in full_isotopologues_names if any(metabolite in iso for metabolite in metabolites)])
        shortened_isotopologues = full_isotopologues[[iso for iso in full_isotopologues_names if any(metabolite in iso for metabolite in metabolites)]]
        
        return shortened_metabolites, shortened_isotopologues

    # <==================================================== BLACK BOX FEATURE EVAL ===================================================================>
    # =================================================================================================================================================

    # =================================================================================================================================================
    # <====================================================== MORANS PROCESSING ======================================================================>
    def load_morans_data(self):
        '''
        Loads in morans data from a text file into a dataframe with columns [filename, metabolite, morans_score]
        '''

        # Initialize lists to store data
        files = []
        metabolites = []
        morans_scores = []

        # Open the text file for reading
        with open(self.morans_path, 'r') as file:
            lines = file.readlines()
            i = 0

            if self.tracer == 'BG':
                # This loop only works if a txt file contains info for only a single tracer/tissue
                print("In Glucose Tracing Loop")
                while i < len(lines):
                    # Extract the file name
                    file_name = lines[i].strip()[-9:]
                    
                    # Skip the "Metab Names:" line
                    i += 2
                
                    # Extract metabolite names
                    metab_names = eval(lines[i])
                    
                    # Skip the "Morans Vals:" line
                    i += 2
                    
                    # Extract Morans values
                    morans_vals = eval(lines[i])
                    
                    # Determine the number of relevant metabolites
                    num_metabolites = min(len(metab_names), len(morans_vals))

                    i += 4
                    
                    # Create data for the current file and append to lists
                    for j in range(num_metabolites):
                        files.append(file_name)
                        metabolites.append(metab_names[j])
                        morans_scores.append(morans_vals[j])
                    
                    # Move to the next set of data
                    i += 1
            else:
                self.shared_morans_file = True
                valid_iso_lists = []
                # Loop through the filepaths
                for file_path in self.iso_paths:
                    # Split the path by '/'
                    path_parts = file_path.split('/')
                    # Extract the relevant parts and join them
                    new_path = '/'.join(path_parts[4:8]).replace("-isotopolouges-ranks.csv", "\n")
                    # Get the index of the file in the txt
                    path_index = lines.index(new_path)
                    # Extract the list part from the string
                    list_str = lines[path_index+1].split(":")[1].strip()
                    # Convert the string representation of the list to a Python list
                    valid_isotopologues_list = eval(list_str)
                    # print(valid_isotopologues_list)
                    valid_iso_lists.append(valid_isotopologues_list)

                    self.valid_iso_lists = valid_iso_lists

                return valid_iso_lists

        # Create a DataFrame
        data = {'File': files, 'Metabolite': metabolites, 'Morans_score': morans_scores}
        df = pd.DataFrame(data)

        # Sort the DataFrame by the 'metabolite' column
        df = df.sort_values(by='Metabolite')

        # Reset the index to have a clean index after sorting
        df.reset_index(drop=True, inplace=True)

        self.morans_df = df

        return df

    def morans_df_to_dict(self, df = None, num_replicates = None, morans_cutoff = None):
        morans_cutoff = self.morans_cutoff if not morans_cutoff else morans_cutoff
        num_replicates = self.min_replicates if not num_replicates else num_replicates

        # Create an empty dictionary to store the data
        metabolite_dict = {}
        good_iso_list = []

        if self.shared_morans_file == False:
            # Iterate through the DataFrame
            for index, row in self.morans_df.iterrows():
                metabolite = row['Metabolite']
                morans_score = row['Morans_score']
                file_name = row['File']
                
                # Check if the metabolite is already in the dictionary
                if metabolite in metabolite_dict:
                    # Append the Morans score to the existing list
                    metabolite_dict[metabolite][0].append((file_name, morans_score))
                else:
                    # Create a new list with the Morans score and add it to the dictionary
                    metabolite_dict[metabolite] = ([(file_name, morans_score)], None, None)

            # Reorder the values list based on the specified order
            for key, value in metabolite_dict.items():
                morans_list = value[0]
                average = sum(v for k, v in morans_list) / len(morans_list)
                higher_than_cutoff = sum(1 for k, v in morans_list if v > morans_cutoff) >= num_replicates
                metabolite_dict[key] = (morans_list, average, higher_than_cutoff)
                if higher_than_cutoff:
                    good_iso_list.append(key)

            good_iso_list.sort()
            self.good_isotopologues = good_iso_list

        else:
            valid_iso_names = []
            # Go through each list and map them to the isotopologues in their respective file.
            for i in range(len(self.valid_iso_lists)):
                # Iso Names
                iso_names = list(self.get_data(self.iso_paths[i], full_path=True).columns)
                valid_iso_names_replicate = [iso_names[index] for index in self.valid_iso_lists[i]]
                valid_iso_names.append(valid_iso_names_replicate)

            # Flatten the list of lists into single lists - rework into separate function later
            flat_list = [item for sublist in valid_iso_names for item in sublist]
            flat_list.sort()

            # Returns a dictionary where the keys are the iso indices and the values are the number of times they appear in the flattened list (a count)
            valid_name_dict = dict()
            for metab_name in flat_list:
                if metab_name in valid_name_dict:
                    valid_name_dict[metab_name] += 1
                else:
                    valid_name_dict[metab_name] = 1

            # List of the isos that appear an acceptable amount of times (deemed by cutoff)
            good_iso_list = [index for index in list(valid_name_dict.keys()) if valid_name_dict[index] >= self.min_replicates]
            good_iso_list.sort()


        # Display the dictionary
        return metabolite_dict, good_iso_list

    # <====================================================== MORANS PROCESSING ======================================================================>
    # =================================================================================================================================================

    def scatterplot_replicates(self, predicted, actual, metabolite_name):
        """
        Create a single scatterplot of actual vs predicted values for a given metabolite across 6 replicates.
        
        Args:
            predicted (list of DataFrames): List of DataFrames containing predicted values.
            actual (list of DataFrames): List of DataFrames containing actual values (corresponding to predicted values).
            metabolite_name (str): Name of the metabolite column to be plotted.
        
        Returns:
            None (displays a combined scatterplot).
        """
        marker_size = 10
        buffer_width = 0.1

        # Check if the number of replicates matches
        if len(predicted) != len(actual):
            raise ValueError("Number of replicates in predicted and actual data must be the same.")

        # Create a color palette for the replicates
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        # Create a color map for replicates
        colormap = plt.cm.get_cmap("tab10", len(predicted))


        # Create a single scatterplot
        plt.subplots(figsize=(6, 6))
        plt.title(f'{metabolite_name}')

        for i in range(len(predicted)):
            replicate_name = f'Replicate {i + 1}'
            replicate_color = colors[i % len(colors)]
            # Use the colormap to assign colors based on the replicate index
            #replicate_color = colormap(i)


            # Extract data for the current replicate
            df_predicted = predicted[i]
            df_actual = actual[i]

            # Check if the metabolite exists in the dataframes
            if metabolite_name not in df_predicted.columns or metabolite_name not in df_actual.columns:
                print(replicate_name + " (Metabolite not found)")
                continue

            # Calculate Euclidean distances from the points to the unit line
            euclidean_distances = np.abs(df_actual[metabolite_name] - df_predicted[metabolite_name])

            # Define buffer zones based on buffer_width
            within_buffer = euclidean_distances <= buffer_width

            # Plot points within buffer zones with adjusted transparency
            plt.scatter(
                df_actual[metabolite_name][within_buffer],
                df_predicted[metabolite_name][within_buffer],
                c=replicate_color,
                marker='o',
                edgecolor=replicate_color,  # Outline color
                s=marker_size,  # Marker size (variable)
                alpha=0.0,  # Adjust transparency for points within buffer zones
                label=replicate_name,
            )

            # Plot points outside buffer zones with full transparency
            plt.scatter(
                df_actual[metabolite_name][~within_buffer],
                df_predicted[metabolite_name][~within_buffer],
                c=replicate_color,
                marker='.',
                edgecolor=replicate_color,  # Outline color
                s=marker_size,  # Marker size (variable)
                alpha=1,  # Full transparency for points outside buffer zones
            )


            # Calculate and plot the line of best fit with bolder lines
            m, b = np.polyfit(df_actual[metabolite_name], df_predicted[metabolite_name], 1)
            plt.plot(
                df_actual[metabolite_name],
                m * df_actual[metabolite_name] + b,
                color=replicate_color,
                linestyle='--',
                linewidth=2,  # Adjust line width for prominence
            )

        # Add labels and legend
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'Replicate {i + 1}') for i in range(len(predicted))]
        plt.legend(handles=legend_handles)
        # Set x and y limits between 0 and 1
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.show()


    def structural_similarity_index(self, predicted, actual, metabolite_name, replicate_index = 0):
        # https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

        # Pull the selected metabolite from 
        coords_df = self.coords_df[replicate_index]

        # Setting x and y boundaries for individual plots, shifted down
        x_min = coords_df['x'].min()
        x_max = coords_df['x'].max()
        y_min = coords_df['y'].min()
        y_max = coords_df['y'].max()

        coords_df['x'] = coords_df['x'] - x_min
        coords_df['y'] = coords_df['y'] - y_min

        plotting_df = coords_df[['x', 'y']]
        plotting_df['predicted'] = predicted[replicate_index][metabolite_name]
        plotting_df['actual'] = actual[replicate_index][metabolite_name]

        x_range = x_max - x_min + 1
        y_range = y_max - y_min + 1

        brain_actual = np.zeros((x_range,y_range))
        brain_predicted = np.zeros((x_range,y_range)) 
 
        for index, row in plotting_df.iterrows():
            brain_actual[int(row['x']), int(row['y'])] = row['actual']
            brain_predicted[int(row['x']), int(row['y'])] = row['predicted']

        brain_actual = np.rot90(brain_actual)
        brain_predicted = np.rot90(brain_predicted)

        SSIM = ssim(brain_actual, brain_predicted, data_range=brain_actual.max() - brain_actual.min())

        return SSIM
# ================================================================= ISOLEARNER =================================================================


# ================================================================= BLACK BOX MODEL =================================================================
class NeuralNetRegressorX(BaseEstimator, RegressorMixin):
    '''
    Params for black box model feature: 
        - X: First Variable that will be conditioned on Z. | Shape (N, dim_X) | the heldout metabolite that we are checking the importance of
        - Y: Second Variable that is conditioned on Z. | Shape (N, dim_Y) | all of the targets (isotopolgoues)
        - Z: Conditioned on Variable from X and Y. | Shape (N, dim_Z) | the remaining metabolites (minus the heldout one)
            -> Note that Z can be compressed via processes like PCA to improve runtime speed. 

    - class_reg_on_X: 
    [Class] -> This class should generate objects (functions) have two methods: 
        - class_reg_on_X().fit(Z, X): .fit() should take in two parameters and train a model that will output an expectation of X given a new sample from Z: 
            Z: Domain data (all of the other metabolites)
            X: Conditional Expectation Outcomes (the held out metabolite)

        - class_reg_on_X().predict(Z): .predict() should return all the conditional expectations off of Z based on the model trained with .fit().
            Z: "Testing" Domain Data

    '''
    def __init__(self):
        print("Initiating reg on X")

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            BatchNormalization(),

            tf.keras.layers.Dense(1)  # Output layer
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self, Z, X):
        self.input_dim = Z.shape[1]
        self.model = self.build_model()
        self.model.fit(Z, X, epochs=10, verbose=0)
        return self.model

    def predict(self, Z):
        return self.model.predict(Z, verbose = 0)

class NeuralNetRegressorY(BaseEstimator, RegressorMixin):
    '''
    - class_reg_on_Y: 
    [Class] -> This class should generate objects (functions) have two methods: 
        - class_reg_on_Y().fit(Z, Y): .fit() should take in two parameters and train a model that will output an expectation of Y given a new sample from Z: 
            Z: Domain data 
            Y: Conditional Expectation Outcomes
        - class_reg_on_Y().predict(Z): .predict() should return all the conditional expectations off of Z based on the model trained with .fit().
            Z: "Testing" Domain Data
    '''

    def __init__(self, lambda_val = 0.001):
        self.lambda_val = lambda_val

    def build_model(self):
        model = tf.keras.Sequential([
            # Input Layer
            Dense(128, input_dim = self.input_dim, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            
            Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            
            # Removed relu to allow negative 
            Dense(self.output_dim, kernel_initializer='he_uniform', kernel_regularizer=l2(self.lambda_val))
        ])

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 3e-05),
                    #loss=tf.keras.losses.MeanSquaredError(),
                    loss = tf.keras.losses.MeanSquaredError(),
                    metrics=['mse', 'mae'])        
                    
        return model

    def build_IsoLearner(self):
        model = tf.keras.Sequential([
            # Input Layer
            Dense(128, input_dim = self.input_dim, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            
            Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            
            Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            
            Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            Dropout(0.25),      
            
            Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(self.lambda_val)),
            BatchNormalization(),
            
            # Removed relu to allow negative 
            Dense(self.output_dim, kernel_initializer='he_uniform', kernel_regularizer=l2(self.lambda_val))
        ])

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 3e-05),
                    #loss=tf.keras.losses.MeanSquaredError(),
                    loss = tf.keras.losses.MeanSquaredError(),
                    metrics=['mse', 'mae'])

        return model
        

    def fit(self, Z, Y):
        self.input_dim = Z.shape[1]
        self.output_dim = Y.shape[1]
        # self.model = self.build_model()
        self.model = self.build_IsoLearner()
        self.model.fit(Z, Y, epochs=50, verbose=0)
        return self.model

    def predict(self, Z):
        return self.model.predict(Z, verbose = 0)

def compare_tracers(list_of_tracer_results = [], list_of_tracers = []):
    '''
    This function takes the IsoLearner results across multiple tracers and compiles them to see trends that are generalizable. 
    '''

    metab_dicts = []

    for IsoLearner_object, tracer in zip(list_of_tracer_results, list_of_tracers):
        print(tracer)
        IsoLearner_object.actual, IsoLearner_object.predicted = IsoLearner_object.cross_validation_testing(checkpoints_dir_label = f"{tracer}-Morans-optimal-model")
        metab_dict = IsoLearner_object.cross_validation_eval_metrics(IsoLearner_object.actual, IsoLearner_object.predicted, plot = False)
        # df = IsoLearner_object.print_evaluation_metrics(IsoLearner_object.actual, IsoLearner_object.predicted, num_rows=200, create_df=True, latex_table=False)
        # metab_dict = IsoLearner_object.relative_metabolite_success(isotopologue_metrics = df, all_isotopologues=list(IsoLearner_object.actual.columns), plot = False)
        metab_dicts.append(metab_dict)

    return metab_dicts