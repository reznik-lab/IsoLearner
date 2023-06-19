from ion_count import load_data, tic_normalization, generate_data, convert_to_ranks
from os.path import exists
import seaborn as sns
import pandas as pd 
import numpy as np

if False:
    tracer_dir = 'Brain-15NNH4Cl'
    tracer = 'B15NNH4Cl'
    path1 = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-m0-no-log/{tracer_dir}/{tracer}-FML-KD-M1.csv'
    path2 = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-m0-no-log/{tracer_dir}/{tracer}-FML-KD-M2.csv'
    path3 = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-m0-no-log/{tracer_dir}/{tracer}-FML-KD-M3.csv'
    path4 = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-m0-no-log/{tracer_dir}/{tracer}-FML-ND-M1.csv'
    path5 = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-m0-no-log/{tracer_dir}/{tracer}-FML-ND-M2.csv'
    path6 = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-m0-no-log/{tracer_dir}/{tracer}-FML-ND-M3.csv'
    paths = [path1, path2, path3, path4, path5, path6]
    replicates = ['KD-M1', 'KD-M2', 'KD-M3', 'ND-M1', 'ND-M2', 'ND-M3']

tracers = ["glucose", "lactate", "glutamine", "glycerol", "citrate", "3HB", "acetate"]
replicates = ["M1", "M2", "M3"]

n_dims = 5  # hyperparameter of latent dimensions
seed = 42
start_index = 2
iso_index = 5
sub_dir = 'PCA'
results_dir = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/kidney-m0-no-log'
plot_dir = f'{results_dir}/plots'
perform_rank_transform = False 

for tracer in tracers:
    for replicate in replicates[2:]:
        file_path = f"/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/full-metabolite-list/separated/FML-kidney-{tracer}-{replicate}.csv"
        data, meta_data, iso_names, iso_map, K, metabolite_count, metabolite_map = load_data(file_path, start_index, iso_index, drop_first=False)
        censor_indicator = np.where(data == 0, 1, 0)  # 1 means censored, 0 means not censored
        # TIC normalization - total ion count normalization
        #   - for each metabolite, divide by the sum of that row 
        normalized_data = tic_normalization(data)
        print(normalized_data)

        N, J, Z, L, start_col, stop_col, Y, X = generate_data(normalized_data, K, n_dims)
        
        if perform_rank_transform:
            Y = convert_to_ranks(Y)
            X = convert_to_ranks(X)

        iso_map = {s: i for i, s in enumerate(iso_names)}
        pd.concat([pd.DataFrame(Y, columns=list(metabolite_map.keys())), meta_data],axis=1).to_csv(f'{results_dir}/{tracer}-{replicate}-ioncounts-ranks.csv')
        pd.concat([pd.DataFrame(X, columns=list(iso_names)), meta_data],axis=1).to_csv(f'{results_dir}/{tracer}-{replicate}-isotopolouges-ranks.csv')


if False:
    for i, file_path_replicate in enumerate(zip(paths[3:], replicates[3:])):
        # path = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-m0-no-log/Brain-3HB/B3HB-FML-ND-M{i+1}.csv'

        file_path = file_path_replicate[0]
        replicate = file_path_replicate[1]

        data, meta_data, iso_names, iso_map, K, metabolite_count, metabolite_map = load_data(file_path, start_index, iso_index)
        censor_indicator = np.where(data == 0, 1, 0)  # 1 means censored, 0 means not censored
        # TIC normalization - total ion count normalization
        #   - for each metabolite, divide by the sum of that row 
        normalized_data = tic_normalization(data)
        print(normalized_data)

        N, J, Z, L, start_col, stop_col, Y, X = generate_data(normalized_data, K, n_dims)
        Y = convert_to_ranks(Y)
        X = convert_to_ranks(X)

        iso_map = {s: i for i, s in enumerate(iso_names)}
        pd.concat([pd.DataFrame(Y, columns=list(metabolite_map.keys())), meta_data],axis=1).to_csv(f'{results_dir}/{tracer}-{replicate}-ioncounts-ranks.csv')
        pd.concat([pd.DataFrame(X, columns=list(iso_names)), meta_data],axis=1).to_csv(f'{results_dir}/{tracer}-{replicate}-isotopolouges-ranks.csv')

if False:
    data, meta_data, iso_names, iso_map, K, metabolite_count, metabolite_map = load_data(FML_path, start_index, iso_index)
    censor_indicator = np.where(data == 0, 1, 0)  # 1 means censored, 0 means not censored
    # TIC normalization - total ion count normalization
    #   - for each metabolite, divide by the sum of that row 
    normalized_data = tic_normalization(data)
    # print(normalized_data)

    N, J, Z, L, start_col, stop_col, Y, X = generate_data(normalized_data, K, n_dims)
    #Y = convert_to_ranks(Y)
    #X = convert_to_ranks(X)
    iso_map = {s: i for i, s in enumerate(iso_names)}
    #pd.concat([pd.DataFrame(Y, columns=list(metabolite_map.keys())), meta_data],axis=1).to_csv(f'{results_dir}/BG-{path_var}-FML-ioncounts.csv')
    #pd.concat([pd.DataFrame(X, columns=list(iso_names)), meta_data],axis=1).to_csv(f'{results_dir}/BG-{path_var}-FML-isotopolouges.csv')

