import tensorflow as tf
import isotopolouge_imputer as imputer
import visualization as vis
import IsoLearner
import numpy as np
from black_box_features import *

Brain_3HB_IsoLearner = IsoLearner.IsoLearner(absolute_data_path = "/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data", relative_data_path = "brain-m0-no-log/Brain-3HB")

metabolites = Brain_3HB_IsoLearner.clean_ion_data[0].columns
short_metabs, short_isos = Brain_3HB_IsoLearner.black_box_feature_evaluation(metabolites = metabolites)

# Example Implementation: 
# ===========================================================
print("Starting GCM Calculations") 
num_cols = len(metabolites)
mutation_index = 0
gcm_output = [None] * num_cols # np.fromiter(np.zeros(num_cols), dtype=float)

all_mutations = short_metabs.astype(float)  # The features (metabolites) that are being considered
Y_data = short_isos.astype(float) # The targets (isotopologues)

# all_mutations = expanded_df.iloc[:, 9:].astype(float) # -> All Mutations We are Testing Over
# Y_data = expanded_df.iloc[:,0:8].astype(float)

y_data = Y_data.values # -> Outputs don't change over the various features we are testing. 
print(f'Y data shape: {y_data.shape}')

while mutation_index < num_cols: # -> Per Mutation (feature)
    print(mutation_index)
    X_data = (all_mutations.iloc[:,mutation_index]).values.reshape(-1,1)
    print(X_data.shape)
    
    #Calculate marginal p_values and if any of the columns of Y 
    # If there's zero correlation, we know there will be no dependence --> ask wes about this
    #r, p_value = pearsonr(X_data, y_data) # -> Skip any features or mutations that don't matter
    #if p_value < 0.05: 
    Z_data = all_mutations.drop(all_mutations.columns[mutation_index], axis=1)
    print(Z_data.shape)

    if False:
        # Get just the top 100 PCs as a quick sub for Z -> wanted to reduce calculations, PCA helps prevent overfitting
        pca = PCA(n_components=5) # -> PCA Decomposition to help reduce dimensionality and reduce overfitting by the models. 
        pca.fit(Z_data.T)
        pca_Z_data = pca.components_.T

    test_mutation_results = multi_gcm_test(X_data, y_data, Z_data, IsoLearner.NeuralNetRegressorX, IsoLearner.NeuralNetRegressorY) # -> Calling the actual test mutation method
    gcm_output[mutation_index] = test_mutation_results #p_value -> Saving Pvalue Results
    #else: 
    #    gcm_output[mutation_index] = 1

    if mutation_index % 50 == 0:
        print(mutation_index)
        # if gcm_output[mutation_index] < 1:
        #     print(gcm_output[mutation_index])
    mutation_index += 1

print("Finished")

print(f'test_mutation_results: {test_mutation_results}')
print(f'GCM outputs: {gcm_output}')

file = open('GCM_output.txt','w')
file.writelines(gcm_output)
file.close()