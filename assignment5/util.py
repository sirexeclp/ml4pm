import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import seaborn as sns

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
    
    y_train = (y_train.values == 'M').astype(float)
    y_test = (y_test.values == 'M').astype(float)
    
    if testing_data:
        return x_test, y_test
    else:
        return x_train, y_train
    
    
def plot_confusion_matrix(y_hat, y_test, threshold=0.5):
    
    print("number samples: " + str(y_test.shape[0]))
    print("number M: " + str((y_test).sum()))
    print("number B: " + str((y_test==0.).sum()))

    y_hat = (y_hat >= threshold).astype(int)
    
    cm = pd.DataFrame(confusion_matrix(y_test,y_hat, labels=[0, 1]),index=["B", "M"],columns=["B", "M"])
    sns.heatmap(cm,annot=True,fmt="d")
    plt.xlabel("predicted diagnosis")
    plt.ylabel("true diagnosis")

    print("Sensitivity: {:.3f}".format( cm["M"]["M"] / (cm["M"]["M"]+cm["B"]["M"])))
    print("Specificity: {:.3f}".format( cm["M"]["M"] / (cm["M"]["M"]+cm["M"]["B"])))
