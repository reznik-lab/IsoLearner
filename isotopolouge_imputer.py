import tensorflow as tf
import pandas as pd 
import numpy as np
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Add, Dense, Activation, BatchNormalization, Lambda, ReLU, PReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from visualization import *

def get_data(file_name = "brain-glucose-KD-M1-isotopolouges.csv", dir = "/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data"):
    '''
    Convert file from csv to dataframe and remove unnecessary columns 

    Parameters:
        - file_name: name of the file
        - dir: directory containing the file (exclude trailing forward slash)
    
    Returns:
        - data: dataframe of the data
    '''
    data_path = f'{dir}/{file_name}'
    data = pd.read_csv(data_path)
    data = data.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    return data 

def multiple_regression_model(num_ion_counts, num_isotopolouges, lambda_val):
  model = Sequential([
      Dense(128, input_dim = num_ion_counts, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
      Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
      Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
      BatchNormalization(),
      Dropout(0.25),
      Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
      Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
      Dense(256, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
      BatchNormalization(),
      Dropout(0.25),      
      Dense(128, kernel_initializer='he_uniform', activation='relu',kernel_regularizer=l2(lambda_val)),
      Dense(num_isotopolouges, kernel_initializer='he_uniform', kernel_regularizer=l2(lambda_val))
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
              #loss=tf.keras.losses.MeanSquaredError(),
              loss = tf.keras.losses.MeanSquaredError(),
              metrics=['mse', 'mae'])

  return model

def create_large_data(all_data = True, data_path = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data', primary_name = 'brain-glucose-KD-M1-isotopolouges.csv'):
    '''
    Creates feature and target dataframes consisting of different samples concatenated to each other. Assumes there are 6 total - 3 KD and 3 ND. 

    params: 
        - all_data: Bool for whether or not all 6 samples should be turned into a single training file, or the last sample (ND-M3) should be omitted and instead returned as a
                    separate pair of train/test files for later use.
    '''
    # List containing the file names of the isotopolouge data
    isotopolouges_paths = [f'{data_path}/brain-glucose-KD-M{i+1}-isotopolouges.csv' for i in range(3)]
    isotopolouges_paths.extend([f'{data_path}/brain-glucose-ND-M{i+1}-isotopolouges.csv' for i in range(2)])

    # List containing the file names of the ion count data
    ion_counts_paths = [f'{data_path}/brain-glucose-KD-M{i+1}-ioncounts.csv' for i in range(3)]
    ion_counts_paths.extend([f'{data_path}/brain-glucose-ND-M{i+1}-ioncounts.csv' for i in range(2)])

    # If all_data flag activated, include the final brain sample, otherwise return a test feature and target
    if all_data:
        isotopolouges_paths.append(f'{data_path}/brain-glucose-ND-M3-isotopolouges.csv')
        ion_counts_paths.append(f'{data_path}/brain-glucose-ND-M3-ioncounts.csv')
    else:
        test_features = get_data(f'{data_path}/brain-glucose-ND-M3-ioncounts.csv')
        test_targets = get_data(f'{data_path}/brain-glucose-ND-M3-isotopolouges.csv')

    # Load each dataframe from the list and concatenate them to each other for - isotopolouges
    isotopolouges = pd.read_csv(isotopolouges_paths[0])
    for i, path in enumerate(isotopolouges_paths):
        if i == 0:
            continue 
        data = pd.read_csv(path)
        isotopolouges = pd.concat([isotopolouges, data], ignore_index=True, axis = 0)

    # Load each dataframe from the list and concatenate them to each other for - ions
    ion_counts = pd.read_csv(ion_counts_paths[0])
    for i, path in enumerate(ion_counts_paths):
        if i == 0:
            continue 
        data = pd.read_csv(path)
        ion_counts = pd.concat([ion_counts, data], ignore_index=True, axis = 0)

    # Drop the unneeded columns from both features and targets 
    ion_counts = ion_counts.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    isotopolouges = isotopolouges.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)

    if all_data:
        return ion_counts, isotopolouges
    else:
        return ion_counts, isotopolouges, test_features, test_targets 

def preprocess_data(ion_counts, isotopolouges, ALL_SIX = True):
    '''
    Take in the input dataframes and convert the to numpys for the model
    Returns: 
        - num_ion_counts: the number of features going into the model
        - num_isotopolouges: the number of targets being predicted
    '''
    x = ion_counts.to_numpy()
    y = isotopolouges.to_numpy()
    print(x.shape, y.shape)
    num_ion_counts = x.shape[1]
    num_isotopolouges = y.shape[1]

    if ALL_SIX:
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)

        return num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val, x_test, y_test
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        return num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val

def testing(feature_data, target_data, checkpoint_path = './saved-weights/KD-M1-unranked-dropout/checkpoint', train = False):
    num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(feature_data, target_data)
    #num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val = preprocess_data(feature_data, target_data, ALL_SIX = False)
    isotopolouge_names = list(target_data.columns)

    '''
    ALL_SIX = False # Flag to decide whether to train/test on all the brains or only 1 

    if ALL_SIX:
        # Generate a dataset consisting of all brains concatenated to each others
        ion_counts, isotopolouges, metabolite_names, isotopolouge_names = create_large_data()
        # Convert to numpy and split into train, test, val
        num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(ion_counts, isotopolouges)
    
    # When working with only one brain
    num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val = preprocess_data(ion_counts, isotopolouges, ALL_SIX=False)
    '''

    # define model
    model = multiple_regression_model(num_ion_counts, num_isotopolouges, 0.01)

    # Checkpoints
    # https://keras.io/api/callbacks/model_checkpoint/
    checkpoint_filepath = checkpoint_path # './saved-weights/KD-M1-unranked-dropout/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose = 1,
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_best_only=False,
        save_freq= int(217 * 20))

    # Learning rate scheduler 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose = 1, min_lr=0)

    # fit model
    if train:
        history = model.fit(x_train, y_train, batch_size = 32, verbose=1, validation_data = (x_val, y_val), epochs=100, callbacks=[model_checkpoint_callback, reduce_lr])
    
    else:
        model.load_weights(checkpoint_filepath)
        # evaluate model on test set
        mae = model.evaluate(x_test, y_test, verbose=1)

        prediction = model.predict(x_test)
        #plt.scatter(y_test, prediction)
        # plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-predicting.png')
        #plt.show()

        print(y_test.shape, prediction.shape)
        print(y_test)
        print(prediction)
        plot_individual_isotopolouges_2(y_test, prediction, isotopolouge_names, ranked = True)

        # Train on one brain ranked just to see if it works, try multiple brains 
        # Unranked - train 5 test 1 
        # Ranked - train 5 test 1 
        # Have the model transform across each feature by column  

def correlations():
    brain_K1 = get_data(file_name="brain-glucose-KD-M1-isotopolouges.csv")
    brain_K1_ranks = get_data(file_name="brain-glucose-KD-M1-isotopolouges-ranks.csv")
    
    brain_K2 = get_data(file_name="brain-glucose-KD-M2-isotopolouges.csv")
    brain_K2_ranks = get_data(file_name="brain-glucose-KD-M2-isotopolouges-ranks.csv")

    brain_N1 = get_data(file_name="brain-glucose-ND-M1-isotopolouges.csv")
    brain_N1_ranks = get_data(file_name="brain-glucose-ND-M1-isotopolouges-ranks.csv")

    #corr_heatmap(brain_1)
    #corr_heatmap(brain_1_ranks)

    double_corr_heatmap(brain_K1, brain_K1_ranks, title = "KD Brain 1 Pairwise Corr Coeff")
    double_corr_heatmap(brain_K2, brain_K2_ranks, title = "KD Brain 2 Pairwise Corr Coeff")
    double_corr_heatmap(brain_N1, brain_N1_ranks, title = "N1 Brain 1 Pairwise Corr Coeff")

    plt.show()
    isotopolouge_names = list(brain_K1.columns)

    corr_scatter(brain_K1, isotopolouge_names, 1, 2)


def test_model(feature_data, target_data, train = False):
    # Train/Test/Val Split
    num_ion_counts, num_isotopolouges, x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(feature_data, target_data)
    isotopolouge_names = list(target_data.columns)

    # Instantiate model
    model = multiple_regression_model(num_ion_counts, num_isotopolouges, 0.01)

    # Checkpoints
    # https://keras.io/api/callbacks/model_checkpoint/
    checkpoint_filepath = './saved-weights/M1-unranked/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose = 1,
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_best_only=False,
        save_freq= int(217 * 20))

    # Learning rate scheduler 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose = 1, min_lr=0)

    # fit model
    if train:
        history = model.fit(x_train, y_train, batch_size = 64, verbose=1, validation_data = (x_val, y_val), epochs=100, callbacks=[model_checkpoint_callback, reduce_lr])

    model.load_weights(checkpoint_filepath)
    # evaluate model on test set
    mae = model.evaluate(x_test, y_test, verbose=1)

    prediction = model.predict(x_test)
    plt.scatter(y_test, prediction)
    # plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-predicting.png')
    plt.show()

    print(y_test.shape, prediction.shape)
    print(prediction)
    plot_individual_isotopolouges_2(y_test, prediction, isotopolouge_names)

    # Train on one brain ranked just to see if it works, try multiple brains 
    # Unranked - train 5 test 1 
    # Ranked - train 5 test 1 
    # Have the model transform across each feature by column  

def main():
    path = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-isotopolouges.csv'
    isotopolouges = pd.read_csv(path) 
    isotopolouges = isotopolouges.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    print(isotopolouges)
    isotopolouge_names = list(isotopolouges.columns)

    ion_counts = pd.read_csv('/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-ioncounts.csv')
    ion_counts = ion_counts.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    print(ion_counts)
    metabolite_names = list(ion_counts.columns)

    x = ion_counts.to_numpy()
    y = isotopolouges.to_numpy()
    print(x.shape, y.shape)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)
    num_ion_counts = x.shape[1]
    num_isotopolouges = y.shape[1]
    print(num_ion_counts, num_isotopolouges)
    print(x_train, y_train)


    # define model
    model = multiple_regression(num_ion_counts, num_isotopolouges, 0.01)

    # Checkpoints
    # https://keras.io/api/callbacks/model_checkpoint/
    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose = 1,
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_best_only=False,
        save_freq= int(217 * 10))

    # fit model
    # history = model.fit(x_train, y_train, batch_size = 64, verbose=1, validation_data = (x_val, y_val), epochs=100, callbacks=[model_checkpoint_callback])


    model.load_weights(checkpoint_filepath)
    # evaluate model on test set
    mae = model.evaluate(x_test, y_test, verbose=1)

    prediction = model.predict(x_test)
    #plt.scatter(y_test, prediction)
    # plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-predicting.png')
    #plt.show()

    print(y_test.shape, prediction.shape)

    plot_individual_isotopolouges(y_test, prediction, isotopolouge_names)


'''
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-ioncount-accuracy-2.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M1-ioncount-loss-2.png')

plt.show()
'''

if __name__ == "__main__":
    #main()
    if False:
        brain_K1_ions = get_data(file_name="brain-glucose-KD-M1-ioncounts.csv")
        brain_K1 = get_data(file_name="brain-glucose-KD-M1-isotopolouges.csv")
        print(brain_K1_ions, brain_K1)
        testing(brain_K1_ions, brain_K1, './saved-weights/KD-M1-unranked-dropout/checkpoint', train = True)

    if False: 
        ions, isotopolouges = create_large_data()    
        print(ions, isotopolouges)
        testing(ions, isotopolouges, './saved-weights/all-6-unranked-dropout/checkpoint', train = True)


    #ions = get_data(file_name="brain-glucose-ND-M3-ioncounts.csv")
    #isotopolouges = get_data(file_name="brain-glucose-ND-M3-isotopolouges.csv")
    #testing(ions, isotopolouges, './saved-weights/all-6-unranked-dropout/checkpoint', train = False)
    
    ions, isotopolouges = create_large_data()    
    ions, isotopolouges = create_large_data(all_data=True)    

    #correlations()