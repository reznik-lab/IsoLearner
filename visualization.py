import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import math 
import isotopolouge_imputer as isoimpute

def corr_heatmap(input_df, ax = None, cbar = True):
    '''Takes a dataframe and displays it's pairwise correlation coefficients as a heatmap'''
    corr = input_df.corr()
    # f, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, vmin=-0.6, vmax=0.8, annot=False, cbar=cbar, mask = mask, cmap=cmap, ax = ax)
    
def double_corr_heatmap(data1, data2, title = "Pairwise Correlation Coefficients", t1 = "Regular Data", t2 = "Ranked Data"):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title)

    corr_heatmap(data1, ax=axes[0])
    axes[0].set_title(t1)

    corr_heatmap(data2, ax=axes[1])
    axes[1].set_title(t2)
    plt.show()

def corr_scatter(data1, data2, title = "Brain 1 vs Brain 2", x_title = "Brain 1", y_title = "Brain 2"):
    corr1 = data1.corr()
    corr2 = data2.corr()

    for col in corr1.columns:
        plt.scatter(corr1[col], corr2[col], label=col)

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.legend(loc='best', fontsize=0.5)
    plt.show()

def iso_corr_scatter(data, iso_names, iso1_index, iso2_index):
    corr1 = data.corr()
    sns.scatterplot(x=iso_names[iso1_index], y=iso_names[iso2_index], data=data);
    plt.show()
    
# Plot individual isotopolouges 
def plot_individual_isotopolouges(actual, predicted, names):
    #fig,axes = plt.subplots(4,4, figsize=[12,9])
    #for i in range(4):
        #for j in range(4):
    for i, name in enumerate(names):
        if i == 25:
            break

        plt.subplot(5, 5, i+1)
        plt.title(names[i])
        plt.grid()
        plt.scatter(actual[i, :], predicted[i, :])

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
                        
    plt.show()

def plot_individual_isotopolouges_2(actual, predicted, names, specific_to_plot = None, grid_size = 8, ranked = False):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    # https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data
    fig, axs = plt.subplots(grid_size, grid_size, sharex=False, sharey=False, figsize = (40,40))
    iso_index = 0

    if specific_to_plot is not None:
        iso_indices_to_plot = [names.index(name) for name in specific_to_plot]

    
    for ax in axs.flat:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, color='red', alpha=0.75, zorder=10)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    for i in range(grid_size):
        if iso_index == len(names):
            break  
        for j in range(grid_size):
 
            # axs[i, j].scatter(actual[iso_index, :], predicted[iso_index, :])
            if specific_to_plot is not None:
                axs[i, j].scatter(actual[:, iso_indices_to_plot[iso_index]], predicted[:, iso_indices_to_plot[iso_index]])
                axs[i, j].set_title(names[iso_indices_to_plot[iso_index]])
                axs[i, j].title.set_size(20)
            else:
                axs[i, j].scatter(actual[:, iso_index], predicted[:, iso_index])
                axs[i, j].set_title(names[iso_index])
                axs[i, j].title.set_size(30)
            iso_index += 1

            if iso_index == len(names):
                break

    '''
    for ax in axs.flat:
        ax.set(xlabel='actual', ylabel='predicted')
        if ranked == True:
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

        else:
            ax.set_xticks([-5,0.5,5])
            ax.set_yticks([-5,0.5, 5])
    '''
    

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()

    plt.show()

# Plot a single isotopolouge's actual vs predicted values, based on that isotopolouge's index 
def plot_isotopolouge(actual, predicted, index, dynamic_axes = True, ranked = False):
    """
    Parameters: 
        - actual: dataframe containing the true data. Columns are isotopolouges and rows are observations. 
        - predicted: dataframe containing the predicted data returned by the model
        - index: the column number corresponding to which isotopolouge should be plotted
    """

    names = list(actual.columns)
    isotopolouge_name = names[index]

    # Select column with index position 3 (fourth column) -> df.iloc[:, 3]
    actual_isotopolouge = actual.iloc[:, index]
    predicted_isotopolouge = predicted.iloc[:, index]

    plotting_df = pd.DataFrame()
    plotting_df["actual"] = actual_isotopolouge
    plotting_df["predicted"] = predicted_isotopolouge

    if ranked == False:
        ax_offset = 0.5
    else:
        ax_offset = 0

    ax_upper_lim = max(plotting_df["actual"].max() + ax_offset, plotting_df["predicted"].max() + ax_offset)
    ax_lower_lim = min(plotting_df["actual"].min() - ax_offset, plotting_df["predicted"].min() - ax_offset)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(plotting_df, x = "actual", y = "predicted", ax = ax).set(title=isotopolouge_name)
    
    if dynamic_axes:
        ax.set_xlim(ax_lower_lim, ax_upper_lim)
        ax.set_ylim(ax_lower_lim, ax_upper_lim)
        line = np.linspace(ax_lower_lim, ax_upper_lim, 1000)

    else: 
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        line = np.linspace(-5, 5, 1000)
        
    plt.plot(line, line,'k-') # identity line


    #ax.set_xlim(plotting_df["actual"].min() - ax_offset, plotting_df["actual"].max() + ax_offset)
    #ax.set_ylim(plotting_df["predicted"].min() - ax_offset, plotting_df["predicted"].max() + ax_offset)
    plt.show()

# Correlation matrix between total ion counts and isotopolouge breakdowns, per metabolite 
def ion_count_isotopolouge_corr(ion_counts, isotopolouges, ion_index, iso_start_index, iso_end_index):
    '''Plots pair-wise correlation heatmap for a given metabolite and its isotopolouges.'''

    # List of names of the metabolites and isotopolouges
    metabolite_names = list(ion_counts.columns)
    isotopolouge_names = list(isotopolouges.columns)

    # Given an index into the list of metabolite names, return the start and end indices of all of its corresponding isotopolouges
    isotopolouges_from_metabolite_name = list(filter(lambda x: metabolite_names[ion_index] in x, isotopolouge_names))
    start_index = isotopolouge_names.index(isotopolouges_from_metabolite_name[0])
    end_index = start_index + len(isotopolouges_from_metabolite_name)

    # Isolate the specific column with the metabolite 
    ion_count = ion_counts.iloc[:, ion_index].to_frame()
    # Pull the corresponding isotopolouge data using the indices found above
    isotopolouge = isotopolouges.iloc[:, start_index:end_index]

    # isotopolouge = isotopolouges.iloc[:, iso_start_index:iso_end_index]

    metab = list(ion_count.columns)
    iso_names = list(isotopolouge.columns)
    names = metab + iso_names

    data_for_corr = pd.concat([ion_count, isotopolouge], axis = 1, ignore_index=True, names = metab)
    data_for_corr.columns = names
    
    #corr_heatmap(data_for_corr)
    
    '''Takes a dataframe and displays it's pairwise correlation coefficients as a heatmap'''
    corr = data_for_corr.corr()
    # f, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, vmin=-0.6, vmax=0.8, annot=False, cbar=True, mask = mask, cmap=cmap, ax = None).set(title=metabolite_names[ion_index])

    plt.show()


def median_rho_feature_plot(data):
    bar_df = (data
          .sort_values(by=["median_rho"], ascending=False)
          )
    plt.rcParams['figure.figsize'] = [20, 18]
    plt.bar(bar_df["isotopologue"], bar_df["median_rho"], color=bar_df["color"])
    plt.xlabel("isotopologue")
    plt.ylabel("median rho")
    plt.title("median rho for isotopolouges")
    plt.xticks(rotation=-90)
    plt.margins(x=0.01)
    plt.show()
    #plt.savefig(f'{plot_dir}/Isotopologue distribution of significant prediction', format='pdf')
    #plt.close()


def plot_brain(brain_data, iso_index = 0, iso_name = None, normalize = False, cmin=0, cmax = 1, tracer = 'Glucose'):
    '''
    Plots the data for a single isotopolouge as an image (similar to how it would be displayed on IsoScope)
    https://www.geeksforgeeks.org/matplotlib-pyplot-pcolormesh-in-python/

    '''
    metabolite_names = list(brain_data.keys())
    if iso_name != None:
        iso_index = metabolite_names.index(iso_name) 

    iso_to_plot = metabolite_names[iso_index]

    if normalize:
        brain_data[iso_to_plot] = (brain_data[iso_to_plot] - brain_data[iso_to_plot].mean()) / brain_data[iso_to_plot].std() 


    plotting_df = brain_data[[iso_to_plot, 'x', 'y']]

    x_min = plotting_df['x'].min()
    x_max = plotting_df['x'].max()
    y_min = plotting_df['y'].min()
    y_max = plotting_df['y'].max()

    pd.options.mode.chained_assignment = None

    plotting_df['x'] = plotting_df['x'] - x_min
    plotting_df['y'] = plotting_df['y'] - y_min

    pd.options.mode.chained_assignment = 'warn'

    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    brain = np.zeros((x_range,y_range))
    for index, row in plotting_df.iterrows():
        # print(int(row['x']), int(row['y']))
        brain[int(row['x']), int(row['y'])] = row[iso_to_plot]

    brain = np.rot90(brain)


    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    black = np.array([0/256, 0/256, 0/256, 1])

    newcolors[:25, :] = black
    newcmp = ListedColormap(newcolors)

    plt.figure(figsize=(5, 4))

    c = plt.pcolormesh(brain, cmap = newcmp)
    plt.clim(cmin,cmax)

    plt.title(f'{tracer} labeled {iso_to_plot}', fontsize = 'x-large', fontweight ="bold")
    plt.colorbar(c)
    plt.show()


def plot_multiple_brains(brain_data, title = 'Plotting Metabolites', indices_to_plot = [], cmin=0, cmax = 1):
    '''
    Plot a grid of brain images given a dataframe containing the data. Each column in the dataframe represents a 
    metabolite/isotopolouge/thing-to-be-graphed and each row represents an individual observation/pixel. There must be two 
    columns labeled 'x' and 'y' in order to properly graph the iamges. 
        -   The x and y columns should be at the end of the dataframe
    '''
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    # https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data

    # Number of metabolite subplots to make. Subtract 2 to account for the x and y coordinates
    # If a list of isotopolouge indices was provided, use the length of that list instead 
    num_metab_to_plot = len(list(brain_data.keys())) - 2 if (not indices_to_plot) else len(indices_to_plot)

    # Generate the number of rows needed to efficiently house num_metabs
    num_columns = 5
    num_rows = math.ceil(num_metab_to_plot/num_columns)

    iso_num = 0
    isotopolouge_names = list(brain_data.keys())

    # Setting x and y boundaries for individual plots, shifted down
    x_min = brain_data['x'].min()
    x_max = brain_data['x'].max()
    y_min = brain_data['y'].min()
    y_max = brain_data['y'].max()

    brain_data['x'] = brain_data['x'] - x_min
    brain_data['y'] = brain_data['y'] - y_min
    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    # Defining custom Color Map to set a reasonable background color
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    black = np.array([0/256, 0/256, 0/256, 1])

    newcolors[:25, :] = black
    newcmp = ListedColormap(newcolors)

    fig, axs = plt.subplots(num_rows, num_columns, sharex=False, sharey=False, figsize = (50/4 * num_columns,13*num_rows))
    # ax[x,y].set_visible(False)

    for plt_row in range(num_rows):
        if iso_num == num_metab_to_plot:
            break

        for plt_column in range(num_columns):
            # Set the remaining unused spaces to blank slots instead of empty plots    
            if iso_num >= num_metab_to_plot:
                axs[plt_row, plt_column].set_visible(False)
                continue

            iso_to_plot = isotopolouge_names[iso_num] if (not indices_to_plot) else isotopolouge_names[indices_to_plot[iso_num]]
            plotting_df = brain_data[[iso_to_plot, 'x', 'y']]

            brain = np.zeros((x_range,y_range)) 
            for index, row in plotting_df.iterrows():
                brain[int(row['x']), int(row['y'])] = row[iso_to_plot]

            brain = np.rot90(brain)


            '''
            norm = np.linalg.norm(brain)        # To find the norm of the array
            brain_to_plot = brain/norm          # Formula used to perform array normalization
            print(norm, brain_to_plot)
            '''
            max_index = brain.argmax()
            max_tuple = np.unravel_index(max_index, brain.shape)
            # print(brain[max_tuple])
            
            c = axs[plt_row, plt_column].pcolormesh(brain, cmap = newcmp, vmin = cmin, vmax = cmax)
            # axs[plt_row, plt_column].clim(cmin,cmax)

            axs[plt_row, plt_column].set_title(f'{iso_to_plot}', fontsize = 35, fontweight ="bold")
            iso_num += 1

    # fig.suptitle(title, size = 32)
    plt.show()


def stacked_bar_plot(metabs_success_dict, num_bars = 10):
    metab_names = list(metabs_success_dict.keys())
    successful_metabs_nums = np.array([metabs_success_dict[metab_names[i]][0] for i in range(len(metab_names))])
    unsuccessful_metabs_nums = np.array([metabs_success_dict[metab_names[i]][1] for i in range(len(metab_names))])
    total_metabs_nums = np.array([metabs_success_dict[metab_names[i]][2] for i in range(len(metab_names))])
    removed = total_metabs_nums - successful_metabs_nums - unsuccessful_metabs_nums
        
    metabolites = tuple(i for i in metab_names[0:num_bars])

    weight_counts = {
        "Successfully predicted": successful_metabs_nums[0:num_bars],
        "Not predicted well": unsuccessful_metabs_nums[0:num_bars],
        # "Removed during Moran's I": removed[0:num_bars]
    }

    width = 0.5

    fig, ax = plt.subplots(figsize = (10,10))
    bottom = np.zeros(num_bars)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(metabolites, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("Ratio of isotopologues successfully predicted per metabolite")
    
    plt.xlabel("Metabolites", fontsize = 20)
    plt.ylabel("Num of Isotopologues", fontsize = 20)
    ax.legend(loc="upper right")
    plt.xticks(rotation=90)
    ax.title.set_size(20)

    plt.show()


    # Bar 1: Removed Isotopologues vs Valid Isotopologues
    # Bar 2: Successfully predicted vs not sucessfully predicted
    valid_vs_invalid = np.array([np.sum(successful_metabs_nums) + np.sum(unsuccessful_metabs_nums), np.sum(removed)])
    series = pd.Series(valid_vs_invalid, index=['Valid', 'Invalid'], name='Total Isotopologues')
    succesful_vs_unsuccessful = np.array([np.sum(successful_metabs_nums), np.sum(unsuccessful_metabs_nums)])
    series2 = pd.Series(succesful_vs_unsuccessful, index=['Sucessfully predicted', 'Unsuccessfully predicted'], name='Valid Isotopologues')

    # Initialise the subplot function using number of rows and columns
    
    # figure, axis = plt.subplots(1, 2)
    pd.DataFrame(series).T.plot.bar(rot = 0, stacked=True, figsize = (5, 7), color={"Valid": "blue", "Invalid": "green"})
    pd.DataFrame(series2).T.plot.bar(rot = 0, stacked=True, figsize = (5, 7), color={"Sucessfully predicted": "blue", "Unsuccessfully predicted": "red"})

def plot_ground_vs_pred(ground_truth_df, predicted_df, coords_df, title = 'Plotting Metabolites', indices_to_plot = [], iso_names_to_plot = [], cmin=0, cmax = 1):
    ground_truth_df = ground_truth_df.drop(labels = ['x', 'y'], axis = 1, errors = 'ignore')
    predicted_df = predicted_df.drop(labels = ['x', 'y'], axis = 1, errors = 'ignore')

    iso_num = 0
    isotopolouge_names = list(ground_truth_df.keys())

    shrinking_factor = 4
    if iso_names_to_plot:
        indices_to_plot = [isotopolouge_names.index(iso_name) for iso_name in iso_names_to_plot]


    # Number of metabolite subplots to make. Subtract 2 to account for the x and y coordinates
    # If a list of isotopolouge indices was provided, use the length of that list instead 
    num_metab_to_plot = len(indices_to_plot) * 2

    # Generate the number of rows and columns needed to efficiently house num_metabs
    num_columns = 2
    num_rows = math.ceil(num_metab_to_plot/num_columns)

    color_map = get_colormap()

    # Setting x and y boundaries for individual plots, shifted down
    x_min = coords_df['x'].min()
    x_max = coords_df['x'].max()
    y_min = coords_df['y'].min()
    y_max = coords_df['y'].max()

    coords_df['x'] = coords_df['x'] - x_min
    coords_df['y'] = coords_df['y'] - y_min

    ground_truth_df[['x', 'y']] = coords_df[['x', 'y']]
    predicted_df[['x', 'y']] = coords_df[['x', 'y']]

    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    fig, axs = plt.subplots(num_rows, num_columns, sharex=False, sharey=False, figsize = (50/(4 * shrinking_factor) * num_columns,13/(shrinking_factor) *num_rows))
    for plt_row in range(num_rows):
        if iso_num == num_metab_to_plot:
            break

        for plt_column in range(num_columns):
            # Set the remaining unused spaces to blank slots instead of empty plots    
            if iso_num >= num_metab_to_plot:
                axs[plt_row, plt_column].set_visible(False)
                continue
            
            iso_to_plot = isotopolouge_names[math.floor(iso_num/2)] if (not indices_to_plot) else isotopolouge_names[indices_to_plot[math.floor(iso_num/2)]]

            plotting_df = ground_truth_df[[iso_to_plot, 'x', 'y']] if plt_column == 0 else predicted_df[[iso_to_plot, 'x', 'y']]

            brain = np.zeros((x_range,y_range)) 
            for index, row in plotting_df.iterrows():
                brain[int(row['x']), int(row['y'])] = row[iso_to_plot]

            brain = np.rot90(brain)

            max_index = brain.argmax()
            max_tuple = np.unravel_index(max_index, brain.shape)
            
            c = axs[plt_row, plt_column].pcolormesh(brain, cmap = color_map, vmin = cmin, vmax = cmax)

            axs[plt_row, plt_column].set_title(f'{iso_to_plot}', fontsize = 10, fontweight ="bold")
            iso_num += 1

    fig.suptitle(title, size = 20, fontweight ="bold")
    fig.subplots_adjust(top=0.95)
    plt.show()

def get_colormap():
    # Defining custom Color Map to set a reasonable background color
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    black = np.array([0/256, 0/256, 0/256, 1])

    newcolors[:25, :] = black
    newcmp = ListedColormap(newcolors)

    return newcmp

def cross_validation_results(ground_replicates, predicted_replicates, coords_df = [], iso_to_plot = "", cmin=0, cmax = 1, limited = False):
    '''
    For a specific isotopologue, plot the ground truth vs predicted for each replicate.
    '''
    # If a list of dataframes with coordinates are not provided, generate the filepath list and pull the data ourself
    if not coords_df:
        print("Generating coord files")
        _, coords_dfs_paths = isoimpute.generate_filepath_list(data_path = '/brain-m0-no-log', FML = True, tracer = 'BG')
        coords_df = [isoimpute.get_data(file_name=f"{path}", keep_coord=True).loc[:, ['x', 'y']] for path in coords_dfs_paths]

    coord_ranges = []
    for i in range(len(ground_replicates)):
        # Setting x and y boundaries for individual plots, shifted down
        x_min = coords_df[i]['x'].min()
        x_max = coords_df[i]['x'].max()
        y_min = coords_df[i]['y'].min()
        y_max = coords_df[i]['y'].max()

        coords_df[i]['x'] = coords_df[i]['x'] - x_min
        coords_df[i]['y'] = coords_df[i]['y'] - y_min
        
        ground_replicates[i][['x', 'y']] = coords_df[i][['x', 'y']]
        predicted_replicates[i][['x', 'y']] = coords_df[i][['x', 'y']]

        x_range = x_max - x_min + 1
        y_range = y_max - y_min + 1
        coord_ranges.append([x_range, y_range])

    # Number of metabolite subplots to make. There will be two subplots per replicate, so 2 * # Replicates
    num_metab_to_plot = len(ground_replicates) * 2
    # Names of metabolites
    metabolite_names = list(ground_replicates[0].columns)
    # Index of the metabolite to plot
    # iso_to_plot = metabolite_names.index(iso_name)

    # Generate the number of rows and columns needed to efficiently house num_metabs
    num_columns = 2
    num_rows = math.ceil(num_metab_to_plot/num_columns)

    color_map = get_colormap()

    iso_num = 0
    shrinking_factor = 4 

    fig, axs = plt.subplots(num_rows, num_columns, sharex=False, sharey=False, figsize = (50/(4 * shrinking_factor) * num_columns,13/(shrinking_factor) * num_rows))
    for plt_row in range(num_rows):
        if iso_num == num_metab_to_plot:
            break

        for plt_column in range(num_columns):
            # Set the remaining unused spaces to blank slots instead of empty plots    
            if iso_num >= num_metab_to_plot:
                axs[plt_row, plt_column].set_visible(False)
                continue
            
            plotting_df = ground_replicates[plt_row][[iso_to_plot, 'x', 'y']] if plt_column == 0 else predicted_replicates[plt_row][[iso_to_plot, 'x', 'y']]
            
            brain = np.zeros((coord_ranges[plt_row][0],coord_ranges[plt_row][1])) 
            for index, row in plotting_df.iterrows():
                brain[int(row['x']), int(row['y'])] = row[iso_to_plot]

            brain = np.rot90(brain)

            max_index = brain.argmax()
            max_tuple = np.unravel_index(max_index, brain.shape)
            
            c = axs[plt_row, plt_column].pcolormesh(brain, cmap = color_map, vmin = cmin, vmax = cmax)

            if plt_column == 0:
                axs[plt_row, plt_column].set_title(f'Replicate {plt_row + 1} Actual', fontsize = 10, fontweight ="bold")
            else:
                axs[plt_row, plt_column].set_title(f'Replicate {plt_row + 1} Predicted', fontsize = 10, fontweight ="bold")
            iso_num += 1

    top_size = 0.9 if limited else 0.95
    fig.suptitle(iso_to_plot, fontsize = 20, fontweight ="bold")
    fig.subplots_adjust(top=top_size)

    plt.show()