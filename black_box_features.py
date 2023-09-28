import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA

'''
This Python code defines a set of functions that are used for analyzing the conditional independence of variables within a given dataset. 
It aims to assess whether two variables, X and Y, are conditionally independent when conditioned on a third variable Z. The code calculates 
various statistical metrics and p-values to determine the degree of dependence between X and Y after taking Z into account.

The functions provided in the code include:

sigma_jklm: Calculates a specific value in the initializing covariance matrix Sigma based on given variables.
get_p_value: Calculates a p-value for a given set of samples and a specific value.
covariance_matrix: Generates a covariance matrix based on a given array of residuals.
univariate_covariance_metric: Computes a covariance metric for univariate variables X and Y.
multivariate_covariance_metric: Computes covariance metrics for multivariate variables X and Y.
multi_gcm_test: Implements a test for generalized covariance metric (GCM) to assess conditional independence between X and Y given Z.

The code aims to determine whether variables X and Y are conditionally independent when Z is taken into consideration. 
It does this by calculating statistics, p-values, and covariance matrices. It also provides example implementations using various 
regression models to test for conditional independence in a real-world scenario.
'''

def sigma_jklm(R_jk, R_lm): #Generating a specific value in the initalizing covariance matrix Sigma
    n = R_jk.size
    #numerator
    numerator = (R_jk.T @ R_lm)/n - (np.mean(R_jk) * np.mean(R_lm))
    
    #denominator
    denominator_1 = np.sqrt(np.square(np.linalg.norm(R_jk))/n - np.square(np.mean(R_jk)))
    denominator_2 = np.sqrt(np.square(np.linalg.norm(R_lm))/n - np.square(np.mean(R_lm)))
    
    return numerator/(denominator_1 * denominator_2) 

def get_p_value(samples, S_n):
    mean = np.mean(samples)
    std_dev = np.std(samples)
    z_score = (S_n - mean) / std_dev

    p_value = 1 - norm.cdf(z_score)
    return p_value

def covariance_matrix(residual_array):
    dim_x = residual_array.shape[0]
    dim_y = residual_array.shape[1]
    
    sigma_matrix = np.empty((dim_x*dim_y, dim_x*dim_y), dtype=object)
    for j in range(dim_x):
        for k in range(dim_y): 
            for l in range(dim_x):
                for m in range(dim_y):
                    sigma_matrix[dim_x*j+k, dim_x*l+m] = sigma_jklm(residual_array[j,k], residual_array[l,m])
                    
    return sigma_matrix

def univariate_covariance_metric(X, Y, Z, reg_on_X, reg_of_Y):
    n = X.size
    R_total = 0

    R_i = (X - reg_on_X.predict(Z).reshape(-1,)) * (Y - reg_of_Y)
#    R_i = (X - reg_on_X.predict(Z).reshape(-1,)) * (Y - reg_on_Y.predict(Z).reshape(-1,))
    
    numerator = (np.sqrt(n)/n) * np.sum(R_i)

    denom = (np.sum((np.square(R_i)))/n) - (np.square((np.sum(R_i)/n)))
    denomenator = np.sqrt(denom)
    
    return np.abs(numerator/denomenator), R_i

def multivariate_covariance_metric(X, Y, Z, reg_func_X, reg_func_Y):
    '''
    X - 100K x 1
    Y - 100K x 100
    Z - 100K x 2
    '''
    dim_x = X.shape[1] # 1
    dim_y = Y.shape[1] # 100 (for now)

    residual_array = np.empty((dim_x, dim_y), dtype=object) # X by Y array of residual arrays (This is ~ (X, Y, N) array)
    gcm_array = np.empty((dim_x, dim_y), dtype=float) # X by Y array of GCM values with each pairwise dimension univariate GCM.

    uni_gcm_reg_Y = reg_func_Y().fit(Z, Y) #Creating a complete model that tries to predict all the Isotopologues with the one left out metabolite
    full_reg_Y = uni_gcm_reg_Y.predict(Z) #Predict the entire set of output isotopologues beforehand. 
    
    for i in range(dim_x):
        uni_gcm_X = X[:,i].reshape(-1,) 
        uni_gcm_reg_X = reg_func_X().fit(Z, uni_gcm_X)

        for j in range(dim_y):
            if j % 10 == 0:
                print(f"Y is currently {j}") 
            uni_gcm_Y = Y[:, j].reshape(-1,) #Take the specific dimension from the Y output. 
            uni_reg_Y = full_reg_Y[:, j].reshape(-1,) #Take the specific dimension from the PREDICTED Y output
            gcm_array[i,j], residual_array[i,j] = univariate_covariance_metric(uni_gcm_X, uni_gcm_Y, Z, uni_gcm_reg_X, uni_reg_Y)

#            uni_gcm_Y = Y[:, j].reshape(-1,) 
#            uni_gcm_reg_Y = reg_func_Y().fit(Z, uni_gcm_Y)
#            gcm_array[i,j], residual_array[i,j] = univariate_covariance_metric(uni_gcm_X, uni_gcm_Y, Z, uni_gcm_reg_X, uni_gcm_reg_Y)
    
    return gcm_array, residual_array

"""
Note that in the definitions of variables, an example walkthrough is provided for clarity and analogy.

This funciton takes in X, Y, Z data and returns a measure of whether X and Y are conditionally 
independent when conditioned on Z. If X and Y are conditionally independent we would expect 
a large p_value and if they aren't then we'd expect a small p_value to be returned.

Requirements: 
- dim_X * dim_Y > 2

Params: 
- X: First Variable that will be conditioned on Z. | Shape (N, dim_X)
- Y: Second Variable that is conditioned on Z. | Shape (N, dim_Y)
- Z: Conditioned on Variable from X and Y. | Shape (N, dim_Z)
    -> Note that Z can be compressed via processes like PCA to improve runtime speed. 

- class_reg_on_X: 
    [Class] -> This class should generate objects (functions) have two methods: 
        - class_reg_on_X().fit(Z, X): .fit() should take in two parameters and train a model that will output an expectation of X given a new sample from Z: 
            Z: Domain data 
            X: Conditional Expectation Outcomes
        - class_reg_on_X().predict(Z): .predict() should return all the conditional expectations off of Z based on the model trained with .fit().
            Z: "Testing" Domain Data
- class_reg_on_Y: 
    [Class] -> This class should generate objects (functions) have two methods: 
        - class_reg_on_Y().fit(Z, Y): .fit() should take in two parameters and train a model that will output an expectation of Y given a new sample from Z: 
            Z: Domain data 
            Y: Conditional Expectation Outcomes
        - class_reg_on_Y().predict(Z): .predict() should return all the conditional expectations off of Z based on the model trained with .fit().
            Z: "Testing" Domain Data

- stat_aggregation_func: This is a method that takes reduces a series of numbers to a single value based on some statistic per experiment. 
    Default: np.max() which will take the max per series of numbers
    # Note - talk to Wes about using a diff function 

- N: Number of samples to use to determine whether the null hypothesis can be rejected based on what the GCM is. 

(Ex. If our experiment is numerous Cell Lines tested on a specific drug with outcomes recorded, 
  X would be one feature of the Cell Line Feature Description, 
  Y would be the outcomes (multidimensional), 
  and Z would be all cell line features minus X.)
  
Returns: 
 - Tuple:
     [0] -> Sn: Max Univariate GCM from the various combinations of X and Y dimensions
     [1] -> p_value: Calculating a p_value from the S_n value compared to an approximated Quantile function. 
"""
def multi_gcm_test(X, Y, Z, class_reg_on_X, class_reg_on_Y, stat_aggregation_func=np.max, N=1000):
    #Checks
    num_experiments = X.shape[0]
    if not (len(X.shape) == len(Y.shape) and len(Y.shape) == len(Z.shape)): 
        raise Exception("X, Y, and Z are not all the same dimensions")
    if num_experiments != Y.shape[0]:
        raise Exception("X and Y first dimensions are not the same")
    if num_experiments != Z.shape[0]:
        raise Exception("X and Z first dimensions are not the same")
    
    # Generalized Covariance Metric
    gcm_values, residual_array = multivariate_covariance_metric(X, Y, Z, class_reg_on_X, class_reg_on_Y)
    S_n = stat_aggregation_func(gcm_values)
    
    # Get Covariance Matrix - figuring out, how the samples pulled from multivariate matrix look compared to univariate
    dim_x = residual_array.shape[0]
    dim_y = residual_array.shape[1]
    cov = covariance_matrix(residual_array)
    S_hat_samples = np.random.multivariate_normal(np.zeros(dim_x*dim_y), cov, size=N)
    
    # Approximate the Quantile Function by taking the max over all N Samples
    G = np.amax(S_hat_samples, axis=1)
    
    p_value = get_p_value(G, S_n)
    
    # Cannot compared the unscaled S_n values, 
    return gcm_values, S_hat_samples, G, S_n, p_value
    

"""
Example Implementation: 
===========================================================
print("Starting GCM Calculations") 
mutation_index = 0
gcm_output = np.fromiter(np.zeros(num_cols), dtype=float)

all_mutations = expanded_df.iloc[:, 9:].astype(float) -> All Mutations We are Testing Over
Y_data = expanded_df.iloc[:,0:8].astype(float)
y_data = Y_data.values -> Outputs don't change over the various features we are testing. 

while mutation_index < num_cols: -> Per Mutation (feature)
    X_data = (all_mutations.iloc[:,mutation_index]).values.reshape(-1,1)
    
    #Calculate marginal p_values and if any of the columns of Y 
    r, p_value = pearsonr(X_data, viability_summary) -> Skip any features or mutations that don't matter
    if p_value < 0.05: 
        Z_data = all_mutations.drop(all_mutations.columns[mutation_index], axis=1)

        # Get just the top 100 PCs as a quick sub for Z
        pca = PCA(n_components=5) -> PCA Decomposition to help reduce dimensionality and reduce overfitting by the models. 
        pca.fit(Z_data.T)
        pca_Z_data = pca.components_.T

        test_mutation_results = test_mutation(X_data, y_data, pca_Z_data, LinearRegression, LinearRegression) -> Calling the actual test mutation method
        gcm_output[mutation_index] = test_mutation_results[4] #p_value -> Saving Pvalue Results
    else: 
        gcm_output[mutation_index] = 1

    if mutation_index % 50 == 0:
        print(mutation_index)
        if gcm_output[mutation_index] < 1:
            print(gcm_output[mutation_index])
    mutation_index += 1

print("Finished")
===========================================================
"""
