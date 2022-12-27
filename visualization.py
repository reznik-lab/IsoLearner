import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

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

def plot_individual_isotopolouges_2(actual, predicted, names, grid_size = 6, ranked = False):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    # https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data

    fig, axs = plt.subplots(grid_size, grid_size, sharex=True, sharey=True)
    iso_index = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            axs[i, j].scatter(actual[iso_index, :], predicted[iso_index, :])
            axs[i, j].set_title(names[iso_index])
            iso_index += 1

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
        if ranked == True:
            ax.set_xticks([0,0.5,1])
            ax.set_yticks([0,0.5, 1])

        else:
            ax.set_xticks([-5,0.5,5])
            ax.set_yticks([-5,0.5, 5])


    for ax in axs.flat:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()
