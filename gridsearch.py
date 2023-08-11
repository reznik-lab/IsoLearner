#import tensorrt
import tensorflow as tf
import pandas as pd
import importlib
import IsoLearner
import visualization as vis
import model
import numpy as np
import csv 

nodes = {
    'layer_1': [128, 256, 512],
    'layer_2': [128, 256, 512],
    'layer_3': [128, 256, 512],
    'layer_4': [128, 256, 512],
    'layer_5': [128, 256, 512],
    'layer_6': [128, 256, 512],
    'layer_7': [128, 256, 512]
}

activation_functions = ['relu', 'tanh']

learning_rate = 3e-05

dropout = [0.25, 0.5]

GridSearch = model.model(nodes=nodes,
                         activation_functions=activation_functions, 
                         learning_rate=learning_rate, 
                         dropout=dropout,
                         lambda_val=0.001,)


models = GridSearch.build_models()

for i in range(len(models)):
    models[i].save(f'./all-models/model-{i}.h5')


Brain_Glucose_IsoLearner = IsoLearner.IsoLearner(absolute_data_path = "/Users/goldfei/Documents/Iso2/generated-data",
                                            relative_data_path = "brain-m0-no-log",
                                            morans_path = 'valid-metabs-brain-glucose.txt', 
                                            FML=False,
                                            tracer = 'glucose')

print('data shape: ')
for i in range(len(Brain_Glucose_IsoLearner.clean_ion_data)):
    print(Brain_Glucose_IsoLearner.clean_ion_data[i].shape, Brain_Glucose_IsoLearner.clean_iso_data[i].shape)

print(f"GPU: {tf.config.list_physical_devices('GPU')}")

Brain_Glucose_IsoLearner.cross_validation_training()

with open('results-final.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(Brain_Glucose_IsoLearner.all_models)):
        writer.writerow(Brain_Glucose_IsoLearner.all_models[i])