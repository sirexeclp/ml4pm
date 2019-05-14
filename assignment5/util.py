
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def standardize(x):
    return (x - np.mean(x))/ np.std(x)

def load_data(path,testing_data=False, columns=None):
    data = pd.read_csv(path)
    # y has the labels and x holds our features
    y = data["diagnosis"]      # M or B 
    x = data.drop(['diagnosis', 'id','Unnamed: 32'],axis = 1 )
    if columns:
        x = x[columns]
        
    x = x.apply(standardize)
        
    x['bias'] = np.ones(x.shape[0])
    # split data train 70 % and test 30 %
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    
    if testing_data:
        return x_test, y_test
    else:
        return x_train, y_train