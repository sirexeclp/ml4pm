import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image 
import math

def load_data(filename, path_to_image_folder, testing_data=False, columns=None):
    data = pd.read_csv(filename)
    
    # Create a dictionary
    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
    }

    # Look up the cell type in the dictionary
    data['cell_type'] = data['dx'].map(lesion_type_dict.get) 

    # Convert the cell_type string in a number
    data['cell_type_idx'] = pd.Categorical(data['cell_type']).codes
    
    #Reduce dataset
        # Too many melanocytic nevi - let's drop out some of them to balance the data a bit more
    data_red = data.drop(data[data.cell_type_idx == 4].iloc[:5000].index)
        #Select only the celltype classes 4,1,5 for this exercise
    data_red = pd.concat([data[data.cell_type_idx == 4], data[data.cell_type_idx == 1], data[data.cell_type_idx == 5] ]).reset_index(drop=True)\
    
    # Create column 'path' with absolute image paths
    path = []
    for row in data_red.iterrows():
        image_id = row[1].image_id
        path_row = os.path.join(path_to_image_folder, image_id + '.jpg')
        path.append(path_row)
    data_red['path'] = path
    
    # Create column with numerical image vector, that is resized to (100, 75) pixels
    #data_red['image'] = data_red['path'].map(lambda x: np.asarray(Image.open(x).resize((28,28))))
    # y has the labels and x holds our features
    y = data_red['cell_type_idx']  
    
    # split data train 70 % and test 30 %
    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(data_red, y, test_size=0.3, random_state=1)

    return x_train_o, x_test_o, y_train_o, y_test_o



# move this to utils.py:
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
