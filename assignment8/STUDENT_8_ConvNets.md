---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Tutorial 8: Convolutional neural networks with Keras


Welcome to the eighth tutorial of the course 'Machine learning for Precision Medicine'.

In this exercise we will work with the colorectral histology MNIST dataset from [kaggle](https://www.kaggle.com/kmader/colorectal-histology-mnist). This data set represents a collection of 4500 textures in histological images (28x28x3) of human colorectal cancer and comprises 8 different tissue classes. 

1) Tumor  
2) Stroma  
3) Complex  
4) Lympho  
5) Debris  
6) Mucosa  
7) Adipose  
8) Empty  

Last week, you learned how to use tensorflow to create your classification model. This week we will use Keras, which is built on top of tensorflow and makes our lives a lot easier. 
<!-- #endregion -->

**Note:**  
We created this notebook on MacOS with `keras 2.1.6`, `tensorflow 1.6.0` and `python 3.5.5`. You should be able to use newer versions. However the **MacOS anaconda release of tensorflow with python 3.7 is broken** and you will not be able to use `Dropout` or some other layers. You won't need those layers for the main part of the exercise, but you might want to use them for the competition. 

```python
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, GlobalMaxPool2D, Dropout
from keras.losses import categorical_crossentropy
from keras.models import Sequential

from keras.optimizers import Adam
from keras import regularizers

from sklearn.model_selection import train_test_split
```

## Preprocessing  the data

We did this already a bit for you. Basically kaggle provides a table that has all 28x28 pixels in columns and a label column from 1-8. We have applied one-hot-encoding to the labels, i.e. 1: [1,0,0,0,0,0,0,0] 2: [0,1,0,0,0,0,0,0] and stacked all flattened image vectors on top of each other. This time we will split the data into training, test and validation set. We are providing you the X_train, Y_train, and X_test arrays. Now we split X_train and Y_train again into training and validation sets. 

```python
X = np.load('X_train.npy')
Y = np.load('Y_train.npy')
print('X.shape = {}'.format(X.shape))
print('Y.shape = {}'.format(Y.shape))
```

```python
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.15, random_state=1)
```

## Introduction to Keras

Keras greatly simplifies the task of building and training deep neural networks. In this exercise you will learn how to build a feed forward and a convolutional neural network using the keras Sequential API. You will then be free to explore regularization and other ways to improve your model.

Once you're satisfied with your performance, you can enter our competition and compete against the other teams. The best teams will be announced one week after the hand-in date.

### Layers

In Keras the main building blocks you will need to build a model are [Keras layers](https://keras.io/layers/about-keras-layers/). Layers implement many of the most common operations performed in neural networks. The beauty of Keras is that you only have to define the shape of your input and the architecture (layer hyper-parameters and activation functions), and the correct shapes for all of the weight matrices and the output etc. will automatically be inferred.

In this exercise we will look specifically at `Dense`, `Conv2D`, `MaxPooling`. We will also introduce the utility layer `Flatten` and the regularization layer `Dropout`.

### The Sequential API

As mentioned above, we build [Keras models](https://keras.io/models/about-keras-models/) using layers. There are two ways to do this. If every layer of your network simply takes the output of the previous layer  and feeds into the next layer you can build your model using the [Sequential API](https://keras.io/models/sequential/). However, more complex architectures with different connectivity patterns require the use of the [Functional API](https://keras.io/models/model/).

For example, to build a softmax regression classifier with l2-regularization, we could do the following: (Ignore the warning)


```python
ridgemodel = keras.Sequential()
ridgemodel.add(Dense(8, activation='softmax', input_shape=(2352,), kernel_regularizer=regularizers.l2(0.001))) # 2352 input features, 8 output features
```

Before you can train the model, it has to be compiled with an optimizer that will adjust the weights after each round:

```python
ridgemodel.compile(Adam(0.0005), 'categorical_crossentropy', metrics=['accuracy'])
```

You can print the architecture with the `summary()` method:

```python
ridgemodel.summary()
```

Layers can have trainable weights, which we can access with the `get_weights()` method:

```python
# we print the shape of the weight matrix and bias vector:
for w in ridgemodel.layers[-1].get_weights():
    print(w.shape)
```

Keras models are trained with the gradient descent algorithm, for which there are different optimizers available. We can train this model iteratively with minibatch gradient descent. Keras also takes care of this for us:

Now we fit our model, by passing the training data in batches, and train for 100 epochs. 

```python
history = ridgemodel.fit(X_train, y_train, batch_size=128, epochs=100, shuffle=True, validation_data=(X_valid, y_valid), verbose=0)
```

And we write a function to look at our training results

```python
def plothistory(h, metric='acc'):
    
    
    if metric == 'acc':
        plt.title('accuracy')
    else:
        plt.title(metric)
        
    plt.plot(h.history['val_'+metric], label='validation')
    plt.plot(h.history[metric], label='train')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')

```

```python
print('Baseline Performance: {}'.format(np.mean(history.history['val_acc'][-4:])))
```

Generally when working with neural networks it is good practice to establish a baseline performance estimate using a simple method, as we did above with only a softmax regression model. The classifier should have gotten an accuracy of about ~0.52 on the validation set. We will now try to beat this performance with neural networks. 


## Fully Connected Network


Let's implement a fully connected network using the Keras sequential API. In a fully connected network each neuron of a given layer is connected to each neuron in the following layer. A layer that contains a number of neurons which all recieve the same input is called a dense layer, because the neurons are "densely connected". Here we stack several of these layers on top of each other. 

# Task 1: Implement a FCNN 
Create a fully connected model by using the keras functions introduced in the example above. Try using different numbers of layers, i.e. start with 3 Dense layers and try different numbers of nodes (neurons). Think about the activation functions that you will need to use for hidden and output layers. 

Hint: Sequential(), add(Dense()), ...

```python
#STUDENT

def get_fully_connected(input_shape=(2352,), n_classes=8):
    
    model = #your_code
    
    #your_code
    
    return model
```

```python
fcnet = get_fully_connected()
fcnet.summary() #Look at the model architechture and the number of trainable parameters. 
```

<!-- #region -->
Below is the architecture we chose when writing the exercise. Try re-implementing this architecture if your model is not learning. We used ReLU activation functions for every layer except the output layer.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 50)                117650    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                2550      
_________________________________________________________________
dense_4 (Dense)              (None, 50)                2550      
_________________________________________________________________
dense_5 (Dense)              (None, 50)                2550      
_________________________________________________________________
dense_6 (Dense)              (None, 8)                 408       
=================================================================
Total params: 125,708
Trainable params: 125,708
Non-trainable params: 0
_________________________________________________________________
```
<!-- #endregion -->

Now we compile the model using the Adam optimizer with a learning rate of 0.0005. Since we have a multi-class model, we need to use a 'categorical-crossentropy' loss function.

```python
fcnet.compile(Adam(lr=0.0005), 'categorical_crossentropy', metrics=['accuracy'])
```

Let's get the training started...

```python
history = fcnet.fit(X_train, y_train, batch_size=128, epochs=50, shuffle=True, validation_data=(X_valid, y_valid), verbose=1)
```

```python
plothistory(history)
print('FCNN Performance: {}'.format(np.mean(history.history['val_acc'][-4:])))
```

The fully connected network you implemented above should have gotten slighly better performance on the validation set. However, both the baseline model and the fully connected model treat every single color channel of every single input pixel as an independent feature. This doesn't truly reflect the problem we are trying to solve. Convolutional Neural Networks can make use of the local dependencies between pixels and are able learn hierarchical feature representations, which might help improve performance on this dataset. 


## Convolutional Neural Networks


Whereas a fully-connected layer requires a flattened image vector, a convolutional neural network an input shape of (pixels_rows, pixel_columns, color-channel). In our case for one image (28,28,3). 

To do so, we can just reshape the stacked image arrays back to the image dimensions by using np.reshape():


```python
X_train_rs, X_valid_rs = np.reshape(X_train, (-1, 28, 28, 3)),  np.reshape(X_valid, (-1, 28, 28, 3))
X_train_rs.shape
```

** Expected output:** 
(3825, 28, 28, 3)

```python
# plot an example of each class
fig, ax = plt.subplots(4,2, figsize=(8,16), sharex=True, sharey=True)
for i, j in enumerate([6, 1, 3, 28, 23, 4, 5, 0]):
    r = i // 2
    c = i % 2
    ax[r,c].imshow(X_train_rs[j], origin='lower')
    ax[r,c].title.set_text(str(y_train[j]))
```

# Task 2: Implement a CNN


Now we can create a CNN with keras. Convolutional layers, which are often combined with max-pooling operations, can function as powerful feature-extractors. As we go deeper in the network the learned features become more complex.

1) For the implementation, you start again with initializing the Sequential model. Then, you add a convolutional layer, in which you apply a number of filters/kernels (i.e. 32) of i.e. size 3x3 over the input representation with an activation function. The output are the convolved features. 

2) After each Conv layer, we apply MaxPooling to downsample the feature representation again, keeping "dominant features". For each of the regions traversed by the max-pooling filter, we will take the maximum activation of that region for each channel and create a new output tensor with reduced dimensions where each channel is the maximum activatuon of a that channel the region of the input. 

You can repeat and stack these two layers on top of each other as often as you like, or stack multiple convolutional layers without max-pooling operations between them. For predicting the output, the tissue classes, we need to flatten the features (`Flatten`-layer) and add one or more Dense layers on top. 

Now implement a convolutional neural network model. Try different numbers of layers, different number of filters, 1 or two Dense layers before the output etc.. Remember that your last layer always needs to have the shape `(n_classes, )` and use the softmax activation function.


```python
# STUDENT

def get_cnn(input_shape=(28,28,3), n_classes=8):
        
    model = #your_code
        
    #your_code
    
    return model
```

```python
cnn = get_cnn()
cnn.summary()
```

<!-- #region -->
Below is the architecture we chose when writing the exercise. Try re-implementing this architecture if your model is not learning. **Tip:** the number of channels in the output of a convolutional layer equals the number of convolutional filters (=kernels) used in that layer. We used ReLU activation functions for every layer except the output layer. We used 3 by 3 convolutional kernels for the first layer, and 2 by 2 convolutional kernels for the other convolutional layers. We added an additional Dense layer with 16 neurons before the output layer and used "valid" padding for all convolutional layers.

```
_______________________________________________________________________________
Layer (type)                           Output Shape               Param #      
===============================================================================
conv2d_1 (Conv2D)                      (None, 26, 26, 32)         896          
_______________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)         (None, 13, 13, 32)         0            
_______________________________________________________________________________
conv2d_2 (Conv2D)                      (None, 12, 12, 64)         8256         
_______________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)         (None, 6, 6, 64)           0            
_______________________________________________________________________________
conv2d_3 (Conv2D)                      (None, 5, 5, 64)           16448        
_______________________________________________________________________________
flatten_1 (Flatten)                    (None, 1600)               0            
_______________________________________________________________________________
dense_7 (Dense)                        (None, 16)                 25616        
_______________________________________________________________________________
dense_8 (Dense)                        (None, 8)                  136          
===============================================================================
Total params: 51,352
Trainable params: 51,352
Non-trainable params: 0
_______________________________________________________________________________
```
<!-- #endregion -->

```python
cnn.compile(optimizer=Adam(lr=0.0005), loss=categorical_crossentropy, metrics=['accuracy'])
history = cnn.fit(X_train_rs, y_train, epochs=50, batch_size=128, validation_data=(X_valid_rs, y_valid), shuffle=True)

# this can take long to run ( ~2 minutes on a MacBook pro with an i7 and our proposed architecture above)
```

```python
plothistory(history)
print('CNN Performance: {}'.format(np.mean(history.history['val_acc'][-4:])))
```

## A Note on Regularization
If you re-implemented our proposed architecture above, you should have gotten a good performance of ~0.75 - 0.8 on the validation set, depending on how many epochs you trained. Did your model get the same? You can now try to improve the performance by for example adding more convolutional filters to the existing layers or by adding additional layers to the network. When you do this, you will likely encounter over-fitting at some point. To prevent over-fitting you can use regularization. Keras offers multiple ways to do regularization. For example when you instantiate a layer, you can set [l1 and l2 regularization penalties](https://keras.io/regularizers/) by using the `kernel_regularizer`, `bias_regularizer` or `activity_regularizer` arguments. Alternatively, you can use dropout regularization. For this you can use the [`Dropout` layer](https://keras.io/layers/core/#dropout).


# Task 3 - Competition!

Get creative! You have a lot of different layers and hyper-parameters to work with. Can you find a model that outperforms the one we proposed above?

A few notes on the competition:

- Write your code in this notebook
- You have to use Keras
- You are only allowed to use the data we provided (`X_train.npy`, `y_train.npy`)
- Averaging the predictions from multiple separate models (ensemble learning) is not allowed
- You have to submit a solution, even if it's just using the model we proposed above
- The evaluation metric is accuracy

```python
# STUDENT

def get_cnn4Comp(input_shape=(28,28,3), n_classes=8):
        
    model = #your_code
        
    #your_code
    
    return model
```

Once you have trained your final model, please adapt the code-snippets below to create your submission. If you submit any other format than `.npy` and the shape of the array is different than (500, 8), we will not process your submission. **Please also call model.summmary() and paste the text into the text field of your submission!**. As always also submit the notebook itself.

```python
comp_model = get_cnn4Comp()

# Call model.summary() and paste the text into your submission on moodle:
comp_model.summary(line_length=140)
# comp_model.compile()
# comp_model.fit() 
```

```python
# Adapt this code to create the data for your submission
X_competition = np.load('X_test.npy')
X_competition = X_competition.reshape(-1,28,28,3) # reshape the data as we did above, assuming you are using a CNN, most importantly: KEEP THE SAME ORDER Of OBSERVATIONS!

predictions = comp_model.predict(X_competition) # do this after fitting your model of course
np.save('predictions.npy', predictions)
```

Just in case you are wondering: We kept the y_test arraym with the ground truth for ourselves. We will compare your predictions with this ground truth and see which team's predictions were most accurate :-D 

<!-- #region -->
Congratulations, you made it through the eighth tutorial of this course!

# Submitting your assignment

Please rename your notebook under your full name and **submit it on the moodle platform**. If you have problems to do so, you can also send it again to machinelearning.dhc@gmail.com

Please rename the file to 8_ConvNets_<GROUP\>.ipynb and replace <GROUP\> with your group-name. Also rename your predictions array with your groupname.


**Checklist for your submission**  
1) Jupyter notebook    
2) Model summary pasted in the submission text field on moodle  
3) predictions_groupNAME_.npy  


As this is also the first time for us preparing this tutorial, you are welcome to give us feedback to help us improve this tutorial.  

Thank you!  

Jana & Remo
<!-- #endregion -->

```python

```
