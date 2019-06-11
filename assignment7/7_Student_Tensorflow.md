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

# Tutorial 7: Tensorflow and neural networks

Welcome to the seventh tutorial of the course 'Machine learning for Precision Medicine'.

In this exercise we will look at a subset the skin MNIST dataset from [kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000). 

The original datasets includes more than 10,015 images of pigmented lesions with the diagnostic categories: 
- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
- basal cell carcinoma (bcc) 
- benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
- dermatofibroma (df) 
- melanoma (mel)
- melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc)

```python
import pandas as pd
import numpy as np
import os
from glob import glob

%matplotlib inline
import matplotlib.pyplot as plt

from PIL import Image
np.random.seed(123)

import itertools
import utils
from utils import random_mini_batches

import tensorflow as tf
from tensorflow.python.framework import ops
tf.VERSION
```

## Data preprocessing

We have preprocessed the data for you already. i.e. we don't load all images because there are too many for this exercise. We also select only the cell-types basal cell carcinoma, melanoma and melanocytic-nevi. We did this preselection already for you in the utils load_data()-function. 

We have selected some pictures from the three selected classes and printed them in a figure. Have a look at the result. 

```python
from IPython.display import display, Image
display(Image(filename='./image_samples.png'))
```

The original image size was (75, 100, 3). The '3' as the last number of the shape, indicates that we are dealing with RGB-images here. 3 for the 3 color channels red, green and blue. Then we reduced the image resolution to 28x28 pixels (resulting in a shape of (28,28,3)), so that we can easily pass it into our model later. 

To make it easier for you to load the data, we prepared the numpy-vectors X_train, X_test, y_train, y_test already for you. I.e. X_train contains the flattened image vectors of all training images. When you flatten the dimensions (28,28,3) of an image, then you get a flat vector of 2352 from 28\*28\*3. The X_train vector containes these flattened image-vectors stacked from all images. Same procedure was applied for the test vector.

```python
X_train = np.load('./X_train.npy')
print('number of training samples: ', len(X_train))
print('Shape of X_train', X_train.shape)

X_test = np.load('./X_test.npy')
print('number of test samples: ', len(X_test))
print('Shape of X_test', X_test.shape)

y_train = np.load('./y_train.npy')
print('Shape of y_train', y_train.shape)

y_test = np.load('./y_test.npy')
print('Shape of y_test', y_test.shape)
```

We have to transform the vectors for later

```python
X_train = X_train.T
print('Shape of X_train', X_train.shape)

X_test = X_test.T
print('Shape of X_test', X_test.shape)

y_train = y_train.T
print('Shape of y_train', y_train.shape)

y_test = y_test.T
print('Shape of y_test', y_test.shape)
```

Below mentioned image shows how the downsampled image looks like. For plotting, we have to transform the flattened shape back into its pixel and channel shape, as required by the imshow function. Change the number to see different samples.

```python
x = X_train[:,1040].reshape(28,28,3)
plt.imshow(x)
```

# Introduction to Tensorflow


This week's Exercise will give you an introduction to the most popular deep-learning framework [Tensorflow](https://www.tensorflow.org/]), which is developed and maintained by Google.

We will first go over the concepts behind Tensorflow, in particular **Tensors** (variables, constants and placeholders), **initialization**, and building a **computational graph**, which we execute inside a tensorflow session.

Afterwards we implement a simple fully connected network "from scratch" to perform an image classification tasks with a subset of the [skin-cancer MNIST](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) data.



## Tensors and Programming in Tensorflow

Taken from the Tensoflow [documentation](https://www.tensorflow.org/guide/tensors):

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TensorFlow, as the name indicates, is a framework to define and run computations involving tensors. A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.*

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; When writing a TensorFlow program, the main object you manipulate and pass around is the **tf.Tensor**. A tf.Tensor object represents a partially defined computation that will eventually produce a value. TensorFlow programs work by first building a graph of tf.Tensor objects, detailing how each tensor is computed based on the other available tensors and then by running parts of this graph to achieve the desired results.*

Writing and running programs in Tensorflow requires the following steps:

1. Define a set of Tensors (variables, constants or placeholders)
2. Write operations among those Tensors (which will produce other Tensors)
3. Initialize the Tensors you defined in step 1.
4. Create a Session
5. Run the Session. This will run the operations you defined in step 2.

Programming this way in Tensorflow can feel a bit verbose and abstract. For this reason a lot of effort is being made in order to make Tensorflow easier to work with, especially for beginners (see the  [Tensorflow 2.0 alpha](https://www.tensorflow.org/alpha)). We will discuss some of these changes and higher level frameworks like [Keras](https://keras.io/) in later exercises. However, this exercise will be performed in the "classical" Tensorflow way following the steps above using as little abstraction as possible.

Here are two code snippets, that perform all of the steps mentioned above:

```python
# defining tensors a and b (step 1)
a = tf.constant(2.)
b = tf.constant(5.)

x = tf.placeholder(tf.float32, name='x')

# defining operations between them (step 2)
c = tf.multiply(a,x)
d = tf.multiply(b,c)

print(d)
```

**Expected output:**
    Tensor("Mul_1:0", dtype=float32)



When we print `d`, we simply get the non-evaluated Tensor object. This is because we only defined a **computational graph**, but we did not execute it.

In order to run the computations in our graph, we have to create a `tf.Session` and run it:

```python
# we skip initialization (step 3), as we do not need to initialize constants...
# we start a tf.Session (step 4)
with tf.Session() as sess:
    # we run the session to produce 'd' (step 5)
    print(sess.run(d, feed_dict={x: 1.}))
```

**Expected output:**
    10.0


This gives the desired result for `x = 1.`. Note that we used `feed_dict` in order to pass the value for `x`, which we defined to be a placeholder. A placeholder is a Tensor for which we will *feed* a value when we run the session. We can feed in different values for `x` this way:

```python
with tf.Session() as sess:
    # we feed x = 2. instead of x = 1.
    print(sess.run(d, feed_dict={x: 2.}))
```

**Expected output:**
    20.0


The graph we are actually executing here is the default graph:

```python
tf.get_default_graph()
```

If we clear the graph before executing the code above, we will get an error:

```python
tf.reset_default_graph()
with tf.Session() as sess:
    try:
        sess.run(d, feed_dict={x: 2.})
    except RuntimeError as e:
        print(e)
```

**Expected output**  
The Session graph is empty.  Add operations to the graph before calling run().


The graph we defined above made use of constants and placeholders. We can also define variables (as opposed to constants), which can change their values between executions of the graph, and which are typically only initialized once in the beginning. Variables are used for trainable parameters. For example, we could set up a linear model with a single input and output variable like this:

```python
# build the computational graph:
tf.reset_default_graph()

tf.set_random_seed(1)

# define the input:
x = tf.placeholder(tf.float32, name='x')

# define the operations: y_hat = x*w + b
b = tf.get_variable('b', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
w = tf.get_variable('w', shape=[1], dtype=tf.float32, initializer=tf.initializers.random_normal())
z = tf.multiply(x, w)
y_hat = tf.add(z, b)

# the desired output we will compare against:
y = tf.placeholder(tf.float32, name='y')

# the loss (squared error):
loss = tf.square(tf.subtract(y, y_hat))

# the initialization operation:
init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    # initialize the variables:
    sess.run(init)
    
    # get the value for y_hat after random initialization of w and x = 1.
    print('w: {}'.format(sess.run(w)))
    
    # get the value for y_hat after random initialization of w and x = 1.
    print('b: {}'.format(sess.run(b)))
    
    # get the value for y_hat after random initialization of w and x = 1.
    print('y_hat: {}'.format(sess.run(y_hat, feed_dict={x:1.})))
        
    # get the loss for x = 1. and y = 1. 
    print('loss: {}'.format(sess.run(loss, feed_dict={x:1., y:1.})))
    
```

**Expected output**  
w: [-0.12510145]
b: [0.]
y_hat: [-0.12510145]
loss: [1.2658533]
    
(Don't worry if there is a warning coming up)


We see that `w` (our weight variable) was initialized randomly, while `b` (the bias variabe) was initialized with 0. This is because we explicitely set different initializers for the two variables using `initializer= ...`.

Above we simply defined a forward pass through a linear regression model. "Under the hood" Tensorflow has already defined the backward pass. We can optimize the parameters of our model by defining an `optimizer` object. You have to call this object along with the loss when running the tf.Session. When called, it will perform an optimization on the given loss with the chosen method and learning rate.

For instance, if we want to use gradient descent the optimizer would be:

```python
# define the optimizer:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    
    # initialize the variables:
    sess.run(init)
    
    # perform 10 steps of optimization:
    for i in range(10):
        
        # we simply pass the same obsevation (x = 1. and y = 1.) 10 times 
        _, c = sess.run([optimizer, loss], feed_dict={x: 1., y: 1.})
        
        weight = sess.run(w)
        bias = sess.run(b)
        
        print('iteration {:0>2} -> loss: {}, w: {}, b: {}'.format(i+1, c, weight, bias))
    
```

We can see the loss going down and the variables `w` and `b` changing after each iteration. Of course it doesn't make much sense to optimize the parameters for a single observation (`x = 1. , y = 1.`) - this is just an illustrative example. But we now already have everything we need in order to start building and optimizing models with Tensorflow.


## Skin Cancer MNIST prediction 
### Building your first classifier with Tensorflow

We will break down the process of defining your feed-forward neural network in to multiple sub-tasks. Here make use of low-level tensorflow objects in order to implement this model. In a later exercise, we will make use of more abstract objects and operations.

But first, let's have a look at the dataset:


## Task 1:

Complete the function `create_placeholders`. It takes the length of a single observation $\mathbf{x}$ (`n_x`) and the length of the target variable $\mathbf{y}$ ( `n_y` which is equal to the number of classes) as input, and returns two `tf.placeholder` objects.  We will feed batches of observations to these objects during training.

The last dimension, which is the "batch dimension" of these objects and which is defined by the `shape` parameter should be `None`. This will allow tensorflow to automatically infer it when we pass batches of a certain size later.


```python
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector
    n_y -- scalar, number of classes
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
        
    X = #your_code
    Y = #your_code
    
    return X, Y
```

As we did in our last exercise, we define the architecture using a list of dictionaries:

```python
NN_ARCHITECTURE = [
    {"input_dim": 2352, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 3, "activation": "softmax"},
] 
```

<!-- #region -->
## Task 2:

Complete the function `initialize_parameters`. It takes an architecture like the one defined in the cell above, and returns a list of dictionaries `parameters` with the following format:

```
[ {weights: W1, bias: b1},
  {weights: W2, bias: b2},
  ....
  {weights: Wn, bias: bn} ]
```

where `W1, W2, b1, b2, Wn, bn` etc are variables returned by the `tf.get_variable` function for layers 1 - n.

You will also have to specify the correct shapes for `W` and `b`. If you are unsure what the correct shapes are, remind yourself of the operation each layer will perform (and look at the previous exercise):

$$ \mathbf{Z}^{\{n\}} = \mathbf{W}^{\{n\}} \mathbf{A}^{\{n-1\}} + \mathbf{b}^{\{n\}} $$

Again, $\mathbf{b}$ is added via broadcasting.

Use `tf.initializers.he_normal(i)` as the initializer for `W`, and `tf.zeros_initializer()` as the initializer for `b`. The names for the variables should be 'W1', 'W2', ..., 'Wn' and 'b1', 'b2', ..., 'bn' for layers 1 - n. Here we use a initialization method proposed by [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html).
<!-- #endregion -->

```python
def initialize_parameters(architecture):
    
    """
    Creates the variables for the tensorflow session.
    
    Arguments:
    architecture -- list of dictionaries of with layer configurations (input_dim, output_dim, activation)
    
    Returns:
    parameters --  list of dictionaries with layer parameters (weights, bias)
    """
    
    tf.set_random_seed(1)
    
    # initialize the list
    parameters = []
    
    for i, param in enumerate(architecture):
        
        input_dim = #your_code
        output_dim = #your_code
        
        W = tf.get_variable("W{}".format(i+1), [output_dim, input_dim], initializer=tf.initializers.he_normal(i))
        b = tf.get_variable("b{}".format(i+1), [output_dim, 1], initializer=tf.zeros_initializer())
        
        
        layer_parameters = {}
        
        layer_parameters["weights"] = W
        layer_parameters["bias"] = b
        
        parameters.append(layer_parameters)
        
    return parameters

```

Let's create the session here and initialize the parameters for the model architecture. 

```python
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters(NN_ARCHITECTURE)
    
    for i in range(len(NN_ARCHITECTURE)):
    
        print("weights of layer {} = ".format(i+1) + str(parameters[i]["weights"]))
        print("   bias of layer {} = ".format(i+1) + str(parameters[i]["bias"]))

```

**Expected Output:**  
weights of layer 1 = <tf.Variable 'W1:0' shape=(25, 2352) dtype=float32_ref>  
   bias of layer 1 = <tf.Variable 'b1:0' shape=(25, 1) dtype=float32_ref>  
weights of layer 2 = <tf.Variable 'W2:0' shape=(25, 25) dtype=float32_ref>  
   bias of layer 2 = <tf.Variable 'b2:0' shape=(25, 1) dtype=float32_ref>  
weights of layer 3 = <tf.Variable 'W3:0' shape=(25, 25) dtype=float32_ref>  
   bias of layer 3 = <tf.Variable 'b3:0' shape=(25, 1) dtype=float32_ref>  
weights of layer 4 = <tf.Variable 'W4:0' shape=(3, 25) dtype=float32_ref>  
   bias of layer 4 = <tf.Variable 'b4:0' shape=(3, 1) dtype=float32_ref>  



## Task 3:
We now define the forward pass through our model.

Complete the function `forward_propagation`. It takes as input the placeholder tensor `X`, the `architecture` and `parameters` which are defined above. For all layers, these are the operations that have to be performed:

$$  \mathbf{Z}^{\{l\}} = \mathbf{W}^{\{l\}} \mathbf{A}^{\{l-1\}} + \mathbf{b}^{\{l\}} $$
$$  \mathbf{A}^{\{l\}} = \phi^{\{l\}}(\mathbf{Z})  $$

Where $\mathbf{A}^{\{0\}} = \mathbf{X}_t$ and $\mathbf{b}^{\{l\}}$ is added to the matrix $\mathbf{W}^{\{l\}} \mathbf{A}^{\{l-1\}}$ via [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html). The only activation function $\phi$ you will have to handle is ReLU, for which you should use `tf.nn.relu`.

Hint: You will need the tensorflow functions matmul() and add()

```python
def forward_propagation(X, architecture, parameters):
    
    """
    Implements the forward propagation for the model (builds part of the computational graph)
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    architecture -- list of dictionaries, each dictionary contains "input_dim", "output_dim" and "activation" for a single layer
    parameters -- list of dictionaries, each dictionary contains "weights" and "bias" for a single layer

    Returns:
    -- the output of the last LINEAR unit (Zn).
    
    Tips:
     - use tf.nn.relu to apply the ReLU activation function
     
    """
    # forward prop initialization:
    A = X
    
    for i, layer_parameters in enumerate(parameters):
        
        # linear transformation
        Z = #your_code
        
        if i == len(parameters) - 1:
            # return Z if we are in the last layer
            return Z
        else:
            # otherwise apply the activation function
            if architecture[i]['activation'] == 'relu':
                A = #your_code
            else:
                raise NotImplementedError('activation '+architecture[i]['activation']+' not implemented!')
                
```

```python
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(2352, 1)
    parameters = initialize_parameters(NN_ARCHITECTURE)
    Zn = forward_propagation(X, NN_ARCHITECTURE, parameters)
    print("Zn = " + str(Zn))
```

<!-- #region -->
**Expected Output**:  
```
Zn = Tensor("Add_3:0", shape=(3, ?), dtype=float32)
```
<!-- #endregion -->

## Task 4:

Once we have the output $\mathbf{Z}^{\{n\}}$ of the last layer, we will calculate the loss. Here we will implement two cases, corresponding to two activation functions sigmoid or softmax:

1. if the activation function is sigmoid (single class or multi-label), use `tf.nn.sigmoid_cross_entropy_with_logits`
2. if the activation function is softmax (multi class), use `tf.nn.softmax_cross_entropy_with_logits`

Tensorflow implements numerically stable versions of the cross-entropy loss that take the logits (i.e. the values in $\mathbf{Z}^{\{n\}}$) instead of the predicted probabilities as input.

Remember, the loss for a $m$ samples is given my the average accross all samples. Use `tf.reduce_mean` to get the average accross all samples.

```python
def compute_loss(Y, Z, activation='sigmoid'):
    
    """
    Implements the calculation of the loss
    
    Arguments:
    Y -- output dataset placeholder, of shape (output size, number of examples)
    Z -- tensor containing the output of the last hidden layer (without applying the activation function)
    activation -- activation function for the last hidden layer

    Returns:
    loss -- 
    
    Tips:
     - use tf.nn.sigmoid_cross_entropy_with_logits or tf.nn.softmax_cross_entropy_with_logits depending on the activation function
     - use tf.reduce_mean to get the average accross samples 
     
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    if activation == 'sigmoid':
        loss = #your_code
    elif activation == 'softmax':
        loss = #your_code
    else:
        raise ValueError('activation has to be either sigmoid or softmax!')
        
    return loss

```

## Task 5:

We will now bring everything together in the `model` function. Complete the function by using the functions you implemented above:

```python
def model(X_train, Y_train, X_test, Y_test, architecture, learning_rate = 0.0001,
          num_epochs=1000, minibatch_size = 32, print_loss = True):
    
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    
    tf.set_random_seed(1)      # to keep consistent results
    seed = 3                   # to keep consistent results
    
    (n_x, m) = X_train.shape   # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]        # n_y : output size, i.e. the number of classes
    
    loss_history = []          # To keep track of the loss
    
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    
    # Initialize objects for the parameters (weights and biases) with our function from above
    parameters = #your_code
    
    # Forward propagation: Build the forward propagation in the tensorflow graph with our function from above
    Z_n = #your_code
    
    # Loss function: Add loss function to tensorflow graph
    loss = #your_code
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            
            epoch_loss = 0.                            # Defines a loss related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches per epoch
            
            seed = seed + 1
            
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feed_dict should contain a minibatch for (X, Y).
                
                _ , minibatch_loss = sess.run([optimizer, loss], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_loss += minibatch_loss / num_minibatches
            
            # Print the cost every epoch
            if print_loss == True and epoch % 100 == 0:
                print ("Loss after epoch %i: %f" % (epoch, epoch_loss))
            if print_loss == True and epoch % 5 == 0:
                loss_history.append(epoch_loss)
    
    
        # plot the cost
        plt.plot(np.squeeze(loss_history))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        if architecture[-1]["activation"] == "sigmoid":
            correct_prediction = tf.equal(tf.cast(tf.greater_equal(Z_n, tf.constant(0.5)), "float"), Y)
        elif architecture[-1]["activation"] == "softmax":
            correct_prediction = tf.equal(tf.argmax(Z_n), tf.argmax(Y))
            
        # Calculate accuracy on the test set. Hint: use tensorflows reduce-mean() and cast function
        accuracy = #your_code

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
```

```python
model(X_train, y_train, X_test, y_test, NN_ARCHITECTURE)
```

**Question:**  
Play around with the architecture, (i.e. add another layer), learning rate, epochs, ... as far the your computer allows it. Did you find a constellation, that gives a better result? 

```python

```

Congratulations, you made it through the seventh tutorial of this course!

# Submitting your assignment

Please rename your notebook under your full name and **submit it on the moodle platform**. If you have problems to do so, you can also send it again to machinelearning.dhc@gmail.com

Please rename the file to 7_Tensorflow_<GROUP\>.ipynb and replace <GROUP\> with your group-name.

As this is also the first time for us preparing this tutorial, you are welcome to give us feedback to help us improve this tutorial.  

Thank you!  

Jana & Remo

```python

```
