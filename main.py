from ion_count import load_data, tic_normalization, generate_stan_data, convert_to_ranks
from os.path import exists
import seaborn as sns
import pandas as pd 
import numpy as np

path1 = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1.csv'
path2 = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M2.csv'
path3 = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M3.csv'

n_dims = 5  # hyperparameter of latent dimensions

seed = 42
start_index = 2
iso_index = 5
sub_dir = 'PCA'
results_dir = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data'
plot_dir = f'{results_dir}/plots'
n_dims = 5

for i in range(3):
    path = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M{i+1}.csv'

    data, meta_data, iso_names, iso_map, K, metabolite_count, metabolite_map = load_data(path, start_index, iso_index)
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
    pd.concat([pd.DataFrame(Y, columns=list(metabolite_map.keys())), meta_data],axis=1).to_csv(f'{results_dir}/brain-glucose-KD-M{i+1}-ioncounts-ranks.csv')
    pd.concat([pd.DataFrame(X, columns=list(iso_names)), meta_data],axis=1).to_csv(f'{results_dir}/brain-glucose-KD-M{i+1}-isotopolouges-ranks.csv')
