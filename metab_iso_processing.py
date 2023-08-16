import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display
import seaborn as sns
from scipy import stats
from visualization import *
from collections import Counter
from statsmodels.stats.multitest import multipletests

from prettytable import PrettyTable
import os
import re

# ======================================================== MIProcessing ========================================================

class MIProcessing:
    def __init__(self, absolute_data_path, relative_data_path, morans_path = 'valid-metabs-brain-glucose.txt', tracer = 'glucose', FML = False, num_replicates = 6, morans_cutoff = 0.75):
        '''
        Parameters:
        - absolute_data_path (str): Absolute path to the directory containing the file (exclude trailing forward slash)
        - relative_data_path (str): relative path from absolute data directory to the directory containing all of the relevant data files. (Assumes you're already in the primary data directory)
        - morans_path (str): relative path from current directory to text file containing morans information (replicate metabolite names and morans scores) to be read in
        - tracer (string): prefix for the tracer whose data you want to generate [Glucose: BG, 3-Hydroxybutyrate: B3HB | B15NGln, B15NLeu, B15NNH4Cl]
            - Precuror 'B' stands for brain data, 'G' for Glucose
        - FML (bool): flag indicating whether to use the partial metabolite list (19 metabs) or full metabolite list 
        - num_replicates (int): The number of replicates for this tracer -> how many replicates should match for a metabolite to be kept
        - morans_cutoff (float): The moran's I cutoff score for a metabolite to be kept. 
        '''

        self.absolute_data_path = absolute_data_path
        self.relative_data_path = relative_data_path
        self.morans_path = morans_path
        self.tracer = tracer
        self.FML = FML
        self.num_replicates = num_replicates
        self.morans_cutoff = morans_cutoff

        print("Initializing IsoLearner")

        # Generate filepath list

        self.ion_paths, self.iso_paths = self.generate_filepath_list()

        # Generate list of metabolites to be kept for imputation

        #print("Generating List of valid metabolites from Moran's I calculations ", end = "")
        #self.valid_metabolites = self.generate_valid_metabs()
        
        # Returns list of cleaned replicate data
        #print("Cleaning data ", end = "")
        #self.clean_ion_data, self.clean_iso_data, self.new_metabolite_names, self.new_iso_names, self.coords_df = self.preserve_good_metabs()
        #self.all_models = []

    # ============================================== LIST OF FILEPATHS =====================================================================
    def generate_filepath_list(self):
        print('Generate Filepath List Triggered')
        '''
        Returns relative paths of data files as two lists. If sample includes both normal and ketogenic replicates, the ND replicates are first, and then KD. 
            - Example Filename: 'B3HB-KD-M1-FML-ioncounts-ranks.csv'

        Returns: 
            - ion_counts_paths (list): list of filenames with ion_count data
            - isotopologues_paths (list): list of filenames with iso data
        '''
        iso_path = 'FML-isotopolouges-ranks' if self.FML else 'isotopolouges-ranks'
        ion_path = 'FML-ioncounts-ranks' if self.FML else 'ioncounts-ranks'

        # List containing the file names of the isotopolouge data - normal diet mice
        isotopologues_paths = [f'{self.absolute_data_path}/{self.relative_data_path}/{self.tracer}-ND-M{i+1}-{iso_path}.csv' for i in range(3)]
        # List containing the file names of the ion count data - normal diet mice
        ion_counts_paths = [f'{self.absolute_data_path}/{self.relative_data_path}/{self.tracer}-ND-M{i+1}-{ion_path}.csv' for i in range(3)]

        # These two tracers have Ketogenic mice as well, include them in the filepaths
        if self.tracer == 'glucose' or self.tracer == 'B3HB':
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
        print('checking_data_consistency Triggered')
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
        print(f'ion_inconsistencies {ion_inconsistencies}')
        # List of isotopolouges that need to be removed from all files
        iso_inconsistencies = self.identify_inconsistencies(self.iso_paths, show_progress = False)
        print(f'iso_inconsistencies {iso_inconsistencies}')

        return ion_inconsistencies, iso_inconsistencies

    # ============================================== IDENTIFYING DATA INCONSISTENCIES ============================================================
    def identify_inconsistencies(self, list_of_paths, show_progress = True):
        print(f'identify_inconsistencie triffered')
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
    # <============================================== VALID METABS FROM TXT ============================================================>
    def generate_valid_metabs(self): 
        print('generate_valid_metabs triggered')
        '''
        For a given tracer/tissue combination, read in the metabolites and corresponding morans scores of each replicate. Use these to create a list of metabolites 
        that should be kept for model consideration based on the morans cutoff, and identify the isotopologues that need to be removed.
        
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

        for ion_path in self.ion_paths:
            # Convert the path name to match how it appears in the txt file           
            ion_path = ion_path.replace(self.absolute_data_path,'')
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

        good_metabs = self.map_poor_unlabeled_metabolites(metab_names, moransi_scores)
        good_metabs.sort()
        print(f'good_metabs {good_metabs}')

        return good_metabs
    
    def get_all_metabs(self): 
        '''
        For a given tracer/tissue combination, read in the metabolites and corresponding morans scores of each replicate. Use these to create a list of metabolites 
        that should be kept for model consideration based on the morans cutoff, and identify the isotopologues that need to be removed.
        
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

        for ion_path in self.ion_paths:
            # Convert the path name to match how it appears in the txt file           
            ion_path = ion_path.replace(self.absolute_data_path,'')
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

        return metab_names

    def map_poor_unlabeled_metabolites(self, metab_names, moransi_scores):
        print('map_poor_unlabeled_metabolites triggered')
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
            # List of metabolites that must be dropped that are present in this replicate 
            metab_to_drop = [metab for metab in metabolite_names if metab in ion_inconsistencies or not metab in self.valid_metabolites]
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
                iso_to_drop = [iso for iso in iso_names if iso in iso_inconsistencies or not iso[0:-5] in self.valid_metabolites or not iso in good_iso_names]
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

# ======================================================== MIProcessing ========================================================