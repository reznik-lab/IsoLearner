import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

dir = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/'
path1 = '/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M3-ioncounts.csv'

for i in range(3):
    path = f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M{i+1}-isotopolouges.csv'

    data = pd.read_csv(path)
    data = data.drop(labels = ['x', 'y', 'Unnamed: 0'], axis = 1)
    data = data[['Glucose m+01', 'Glucose m+02', 'Glucose m+03', 'Glucose m+04', 'Glucose m+05', 'Glucose m+06']]
    print(data)
    sns.pairplot(data, height = 1)
    plt.savefig(f'/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data/brain-glucose-KD-M{i+1}-glucose-isotopolouges.png')

#plt.show()
