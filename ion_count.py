from time import sleep
from datetime import datetime
import numpy as np
import pandas as pd
import scipy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from numpy.random import default_rng
from scipy.stats import halfnorm
from scipy.stats import truncnorm
from statsmodels import api
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import rankdata
import math
import os
import matplotlib.pyplot as plt
import argparse
import csv
import warnings
import random
from statsmodels.stats.multitest import multipletests
#import cmdstanpy
#from cmdstanpy import CmdStanModel, set_cmdstan_path

#set_cmdstan_path('/rtsess01/juno/home/xiea1/miniconda3/envs/cmdstanpy/bin/cmdstan')

def load_data(raw_data_path, start_index, iso_index):
    data = pd.read_csv(raw_data_path, header=0)
    #  subset data
    #data = data.loc[data['x'] == 92]
    data = data.loc[:,list(data.sum(axis=0)!=0)]

    # drop metabolites that are not labeled at all (only have m+0)
    # Count the number of isotopologues for each metabolite
    def count_drop_metabolites(data, start_index, iso_index):
        # setup initial values
        count = 1
        K = []
        metabolite_count = {}
        for i in range(start_index + 1, data.shape[1]):
            if (data.columns[i])[:-iso_index] == (data.columns[i - 1])[:-iso_index]:
                count += 1
                if i == data.shape[1] - 1:
                    K.append(count)
                    metabolite_count[data.columns[i - 1][:-iso_index]] = count
            else:
                K.append(count)
                metabolite_count[(data.columns[i - 1])[:-iso_index]] = count
                count = 1

        # clean data--drop samples who have zeros in the data (don't need to do this after adding pseudo-count)
        # data = data[~np.any(data == 0, axis=1)]
        metabolite_map = {k: v for v, k in enumerate(metabolite_count.keys())}
        return data, K, metabolite_count, metabolite_map

    data, K, metabolite_count, metabolite_map = count_drop_metabolites(data, start_index, iso_index)

    # collect meta data
    meta_data = data.iloc[:, :start_index]
    data = data.iloc[:, start_index:]
    iso_names = data.columns
    iso_map = {s: i for i, s in enumerate(iso_names)}
    data = data.to_numpy()

    return data, meta_data, iso_names, iso_map, K, metabolite_count, metabolite_map

def tic_normalization(data):
    normalized_data = np.where(data == 0, np.nan, data)
    # replace 0 with half of the minimum value
    normalized_data = np.where(np.isnan(normalized_data), np.nanmin(normalized_data, axis=0)/2, normalized_data)
    normalized_data = normalized_data / np.sum(normalized_data, axis=1, keepdims=True) * 100000
    return normalized_data

def generate_stan_data(data, K, n_dims):
    N = data.shape[0]  # samples
    J = len(K)  # metabolites
    L = n_dims  # embedding dimensions of W and H
    # For dirichlet distribution, k need to be >= 2.
    print(f'Each metabolite has corresponding number of isotopologues respectively:{K}')
    Z = data.shape[1]  # isotopologues

    def count_YX_generator(data, N, J, K):
        start_col = np.zeros(shape=J, dtype=int)
        stop_col = np.zeros(shape=J, dtype=int)
        start = 0
        Y = np.zeros(shape=(N, J))
        X = np.zeros(shape=(N, Z))

        for j in range(J):
            start_col[j] = start
            stop_col[j] = start_col[j] + K[j]
            Y[:, j] = np.sum(data[:, start_col[j]:stop_col[j]], axis=1)
            # ALR Transformation
            X[:, (start_col[j]+1):stop_col[j]] = np.log(data[:, (start_col[j]+1):stop_col[j]] / data[:, start_col[j]].reshape((-1, 1)))
            start = stop_col[j]
        # Log transformation of the total ion counts Y (to make the data look normal)
        # Y = np.log(Y)
        return start_col, stop_col, Y, X

    start_col, stop_col, Y, X = count_YX_generator(data, N, J, K)
    print(start_col, stop_col)

    return N, J, Z, L, start_col, stop_col, Y, X

def convert_to_ranks(data):
    normalizer = len(data) + 1
    # ranks = (1 + data.argsort(axis=0).argsort(axis=0)) / normalizer # smallest sample has rank 0
    ranks = rankdata(data, method='average', axis=0) / normalizer
    return ranks


def main():
    ############################################################## Set parameters
    seed = 42
    start_index = 2
    iso_index = 5
    raw_data_path = 'kidney_data/FML-kidney-glucose-M3.csv'
    sub_dir = 'PCA'
    results_dir = f'results/{sub_dir}'
    plot_dir = f'{results_dir}/plots'
    n_dims = 5

    for dir in [results_dir, plot_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)


    ############################################################## Load Data
    data, meta_data, iso_names, iso_map, K, metabolite_count, metabolite_map = load_data(raw_data_path, start_index, iso_index)
    censor_indicator = np.where(data == 0, 1, 0)  # 1 means censored, 0 means not censored
    # TIC normalization
    normalized_data = tic_normalization(data)
    N, J, Z, L, start_col, stop_col, Y, X = generate_stan_data(normalized_data, K, n_dims)


    Y = np.log(Y)
    Y = convert_to_ranks(Y)
    X = np.delete(X, start_col, axis=1)
    X = convert_to_ranks(X)
    iso_names = np.delete(iso_names, start_col)
    iso_map = {s: i for i, s in enumerate(iso_names)}
    pd.concat([pd.DataFrame(Y, columns=list(metabolite_map.keys())), meta_data],axis=1).to_csv(f'{results_dir}/FML_kidney_glucose_M3_ranked_Y.csv')
    pd.concat([pd.DataFrame(X, columns=list(iso_names)), meta_data],axis=1).to_csv(f'{results_dir}/FML_kidney_glucose_M3_ranked_X.csv')

    ############################################################## cysteine
    cor = np.full((X.shape[1], 2), np.nan)
    for z in range(X.shape[1]):
        cor[z, 0], cor[z, 1] = spearmanr(Y[:, metabolite_map['cysteine']], X[:,z], axis=0)
    cor = pd.DataFrame(cor, columns=['cor', 'pval'], index=iso_names)

    def correct_for_mult(data):
        data['p_adj'] = multipletests(pvals=data['pval'], method="fdr_bh", alpha=0.05)[1]
        return data
    cor = correct_for_mult(cor)
    cor.to_csv(f'{results_dir}/FML_kidney_glucose_M3_ranked_cor_cysteine.csv')




    ############################################################## PCA
    import sklearn
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    # W
    principal_components = pca.fit_transform(normalized_data)
    pd.concat([pd.DataFrame(data=principal_components, columns=
    ['principal component 1', 'principal component 2']), meta_data], axis=1
              ).to_csv(f'{results_dir}/samples_by_components.csv')
    # H
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(normalized_data.T)
    pd.DataFrame(data=principal_components, index=list(metabolite_map.keys())
                 ).to_csv(f'{results_dir}/metabolites_by_components.csv')



if __name__ == "__main__":
    main()
    