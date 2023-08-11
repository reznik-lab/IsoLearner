import pandas as pd 
import numpy as np
import tensorrt # Only for GPU's on Juno Cluster
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
import keras_tuner as kt

from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor # for building the model
from prettytable import PrettyTable
import os
import re

class model:
    def __init__(self, nodes, activation_functions, learning_rate, dropout, lambda_val):
        '''
         Parameters:
            - nodes: the number of nodes in each layer
            - activation_functions: the activation functions for each layer
            - learning_rate: the learning rate for each layer
            - dropout: the dropout
        '''

        self.nodes_layer_1 = nodes['layer_1']
        self.nodes_layer_2 = nodes['layer_2']
        self.nodes_layer_3 = nodes['layer_3']
        self.nodes_layer_4 = nodes['layer_4']
        self.nodes_layer_5 = nodes['layer_5']
        self.nodes_layer_6 = nodes['layer_6']
        self.nodes_layer_7 = nodes['layer_7']


        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.lambda_val = lambda_val

        self.num_ion_counts = 50
        self.num_isotopolouges = 652

    def build_models(self):
        '''
        Builds the model
        Edit for loops to choose which variables you would like to loop through
        '''
        all_models = []
        
        for activation in self.activation_functions:
            for node_1 in self.nodes_layer_1:
                for node_2 in self.nodes_layer_2:
                    for node_3 in self.nodes_layer_3:
                        for node_4 in self.nodes_layer_4:
                            for node_5 in self.nodes_layer_5:
                                for node_6 in self.nodes_layer_6:
                                    for node_7 in self.nodes_layer_7:
                                    
                                        model = Sequential([
                                        # Input Layer
                                        Dense(node_1, input_dim = self.num_ion_counts, kernel_initializer='he_uniform', activation=activation,kernel_regularizer=l2(self.lambda_val)),
                                        BatchNormalization(),

                                        Dense(node_2, kernel_initializer='he_uniform', activation=activation,kernel_regularizer=l2(self.lambda_val)),
                                        BatchNormalization(),
            
                                        Dense(node_3, kernel_initializer='he_uniform', activation=activation,kernel_regularizer=l2(self.lambda_val)),
                                        BatchNormalization(),
                                        Dropout(0.25),
            
                                        Dense(node_4, kernel_initializer='he_uniform', activation=activation,kernel_regularizer=l2(self.lambda_val)),
                                        BatchNormalization(),
            
                                        Dense(node_5, kernel_initializer='he_uniform', activation=activation,kernel_regularizer=l2(self.lambda_val)),
                                        BatchNormalization(),

                                        Dense(node_6, kernel_initializer='he_uniform', activation=activation,kernel_regularizer=l2(self.lambda_val)),
                                        BatchNormalization(),
                                        Dropout(0.25),      
            
                                        Dense(node_7, kernel_initializer='he_uniform', activation=activation,kernel_regularizer=l2(self.lambda_val)),
                                        BatchNormalization(),
            
                                        # Removed relu to allow negative 
                                        Dense(self.num_isotopolouges, kernel_initializer='he_uniform', kernel_regularizer=l2(self.lambda_val))
                                    ])
                                    

                                    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = self.learning_rate),
                                        #loss=tf.keras.losses.MeanSquaredError(),
                                        loss = tf.keras.losses.MeanSquaredError(),
                                        metrics=['mse', 'mae'])
                                    
                                    print(f'generating model # {len(all_models)}')
                                    all_models.append(model)
                                    

        return all_models