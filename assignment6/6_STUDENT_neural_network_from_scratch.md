---
jupyter:
  jupytext:
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.1
---

# Tutorial 6: Neural networks

Welcome to the sixth tutorial of the course 'Machine learning for Precision Medicine'.

In this tutorial we will implement a neural network architecture, which involves the following steps.

1) initialize the weights  
2) Forward Proagation  
    2.1) Perform linear transformation of input  
    2.2) Compute Activations from the linear transformations   
3) calculate the loss  
4) Backpropagation  
5) update weights 

We will use an artificial dataset here, which we want to separate into two classes. Let's generate the and look at the data first...

```python
# Import necessary python modules
import numpy as np
from sklearn.datasets import make_circles
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# we generate a toy-dataset that is not linearly separable:
X, y = make_circles(n_samples=1000, factor=.4, noise=.10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

sns.set_style("whitegrid")
plt.figure(figsize=(8,8))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), s=50, cmap=plt.cm.Spectral, edgecolors='black');
```

From this plot we can see that the data is not linearly separable. So let's use a neural network model to classify the blue from the red data points. Here we will use a neural network, with 4 hidden layers with 25, 50, 50 and 25 units respectively and an output layer of 2 units for our binary classification (red or blue).

```python
NN_ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
   # {"input_dim": 25, "output_dim": 50, "activation": "relu"},
   #{"input_dim": 50, "output_dim": 50, "activation": "relu"},
    #{"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]
```

As discussed in the lecture, each neuron in our neural network will perform a linear transformation of it's input values, which will produce an intermediate value $z$. We apply a non-linear activation function to $z$ in order to get the *activation* $a$, which in turn will be fed to other neurons, until we arrive at the final neuron(s) which constitute the model output. By sequentially performing many of these operations (linear transformation + non-linear activation), we are able to compute very complex non-linear functions of the input variables.

For each neuron, the linear transformation is parameterized by a weight column-vector $\mathbf{w}$ and a bias parameter $b$. Our non-linear activation functions in this exercise will not have any adjustable parameters.

We call a group of neurons which are parameterized in the same way (i.e. their weight vectors have the same length), and perform the same kind of operation on the same input as a *layer*. We can stack the transposed weight vectors of all neurons ${n}$ in the same layer ${l}$ on top of each other to form a weight matrix $\mathbf{W}^{\{l\}}$ and bias vector $\mathbf{b}^{\{l\}}$. Following this definition, the weight matrix of the layer ${l}$, $\mathbf{W}^{\{l\}}$, is an $n^{\{l\}}$ (number of neurons in this layer) by $n^{\{l-1\}}$ (number of neurons in the previous layer) matrix, and $\mathbf{b}^{\{l\}}$ is a vector of vector of length $n^{\{l\}}$.

In this exercise, we will look at a certain class of neural networks called a feed-forward or densely connected neural network. In a densely connected neural network, each neuron of a layer is connected to all neurons of the previous layer, i.e. every neuron in layer ${l}$, will recieve all the output $\mathbf{a}^{\{l-1\}}$, where $\mathbf{a}^{\{l-1\}}$ is the vector that results from concatenating the $n^{\{l-1\}}$ activations of the previous layer $\{a_1,a_2,...,a_{n^{\{l-1\}}}\}$. The first layer ${(l = 1)}$, receives the input ${\mathbf{a}^{\{0\}}} = \mathbf{x}$, where $\mathbf{x}$ is a single observation in our training set $\mathbf{X}$.

We can express the operations happening within a single layer using matrix multiplaction:
$$  \mathbf{z}^{\{l\}} = \mathbf{W}^{\{l\}} \mathbf{a}^{\{l-1\}} + \mathbf{b}^{\{l\}}$$
$$  \mathbf{a}^{\{l\}} = \phi^{\{l\}}(\mathbf{z}^{\{l\}})$$

where $\phi^{\{l\}}$ is the activation function for layer $l$.

Finally, we are not feeding single observation $\mathbf{x}$ to our network, but rather we are processing an entire batch of observations $\mathbf{X}_t$, where $\mathbf{X}_t$ is an $m$ by $i$ matrix, corresponding to $m$ observations $\mathbf{x}^T$ stacked on top of each other, each having $i$ features. 

$$  \mathbf{Z}^{\{l\}} = \mathbf{W}^{\{l\}} \mathbf{A}^{\{l-1\}} + \mathbf{b}^{\{l\}} $$
$$  \mathbf{A}^{\{l\}} = \phi^{\{l\}}(\mathbf{Z})  $$

Where $\mathbf{A}^{\{0\}} = \mathbf{X}_t$ and $\mathbf{b}^{\{l\}}$ is added to the matrix $\mathbf{W}^{\{l\}} \mathbf{A}^{\{l-1\}}$ via [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html), i.e. it is added element-wise to each column. $\mathbf{A}^{\{l\}}$ and $\mathbf{Z}^{\{l\}}$ both have the shape $(n^{\{l\}}, m)$






To start with the implementation, we have to initialize weights across the entire network architechture. How to initialize weights before training is also a big research topic in Deep Learning. Here, we will just use randomly generated numbers. 


### Forward propagation

```python
def init_layers(nn_architecture, seed = 99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        # save everything in a dictionary
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values
```

Now we want to implement the activation functions for our linear transformations. We can activate neurons either with the sigmoid or relu function. 

In the lecture we were introduced to the ReLU activation function, which will use to activate the neurons in the hidden layers:

$$ ReLU(\mathbf{Z}) = max(0,\mathbf{Z}) $$

Our final output layer will use the sigmoid activation function, which was already introduced in the last exercise:

$$ \sigma(\mathbf{Z}) = \frac{1}{1+exp(-\mathbf{Z})} $$

## Task 1:
Implement the sigmoid and relu functions, which take the linear transformation Z as input.

```python
# STUDENT
def sigmoid(Z):
    sig = 1/(1+np.exp(-Z))
    return sig

def relu(Z):
    relu = np.where(Z>0,Z,0)
    return relu
```

Now, we will implement the forward propagtion of a single layer. This function requires the activations of the previous layer stored in A_prev, the weights stored in W_curr and the bias stored in b_curr, as well as an argument which activation function you want to use. 

## Task 2: 
Implement the linear transformation input $\mathbf{Z}$ of the next layer with this function.

```python
#STUDENT

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    #input times weight, add a bias
    Z_curr = W_curr.dot(A_prev)+b_curr
    #activate
    # return of calculated activation A and the intermediate Z matrix
    return globals()[activation](Z_curr), Z_curr
```

We will now implement the forward propagation through the entire network and call the function above for each layer. 
The function here requires our input data $\mathbf{X}$, our initialized weights and biases stored in params_values and the network architecture. The function will output the activation of the last layer, as well as the memory of all activations, weights and biases from the hidden layers below. 

## Task 3:
Call the forward propagation of a single layer.

```python
# STUDENT

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        
        A_curr, Z_curr = single_layer_forward_propagation(A_prev,W_curr,b_curr,activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory


```

We need to compare our output from the output layer $\hat{\mathbf{Y}}$ with the true $\mathbf{Y}$ and calculate the loss, or cost. This is our objective function that we seek to minimize. 

\begin{equation}
J(w,b) =  -\frac{1}{m} \sum_{i=1}^{m}{y log\hat{y}^{(i)} + (1-y^{(i)}) log(1-\hat{y}^{(i)})}
\end{equation}

## Task 4:
Implement the cost, based on the formula above.

```python
# STUDENT

def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -np.mean(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))
    return np.squeeze(cost)
```

Our output layer outputs $\hat{Y}$ with values between 0 and 1 because it applies the sigmoid function. These values correspond to the probability of belonging to class 1. We now have to set a threshold, which defines, that we assign class 1 to a sample that has a value higher than 0.5 and class 0 if smaller or equals 0.5. Afterwards we calculate the accuracy of our predicted labels, by checking how many $\hat{\mathbf{Y}}$ were equals to the true $\mathbf{Y}$

```python
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()
```

### Backward propagation

We have now implemented the full forward propagation through the network. We now wish to implement back propagation in a similar way. This will require us to calculate the partial derivatives of our Loss function with respect to the trainable model parameters.

For a single layer of our neural network, the gradients are calculated according to the following formulae:

$$ \mathbf{dW}^{\{l\}} = \frac{\delta L}{\delta\mathbf{W}^{\{l\}}} = \frac{1}{m} \mathbf{dZ}^{\{l\}} \mathbf{A}^{\{l-1\}T} $$

$$ \mathbf{db}^{\{l\}} = \frac{\delta L}{\delta\mathbf{b}^{\{l\}}} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{dZ}^{\{l\}(i)} $$

$$ \mathbf{dA}^{\{l-1\}} = \frac{\delta L}{\delta\mathbf{A}^{\{l-1\}}} = \mathbf{W}^{\{l\}T} \mathbf{dZ}^{\{l\}} $$

$$ \mathbf{dZ}^{\{l\}} = \mathbf{dA}^{\{l\}} * \phi^{\{l\}'}(\mathbf{Z}^{\{l\}}) $$


We already saw that these formulae make use of the cached values for $\{\mathbf{Z}^{\{1\}}, \mathbf{Z}^{\{2\}}, ..., \mathbf{Z}^{\{n\}}\}$ and $\{\mathbf{A}^{\{1\}}, \mathbf{A}^{\{2\}}, ..., \mathbf{A}^{\{n\}}\}$ calculated during forward propagation. In a first step, let's implement the formula for $\mathbf{dZ}$ for the sigmoid and ReLU activation functions.

## Task 5:

Implement `relu_backward(dA, Z)` and `sigmoid_backward(dA, Z)`. Both functions take a matrix `dA` ($\mathbf{dA}$), *which will be passed during back-propagation* and cached values `Z` ($\mathbf{Z}$), and return $\mathbf{dA} * \phi^{'}(\mathbf{Z})$, where we substitute $\phi^{'}$ with $ReLU^{'}(z)$ or $\sigma^{'}(z)$, performed element-wise for all values in $\mathbf{Z}$, respectively. **Be aware that $*$ here denotes element-wise multiplication, not matrix-multiplication!**

$$ \mathbf{dZ} = \mathbf{dA} * \phi^{'}(\mathbf{Z}) $$

$$  ReLU^{'}(z) =   \begin{equation}
   \begin{cases}
     1, & \text{if}\ z>0 \\
     0, & \text{otherwise}
   \end{cases}
\end{equation} $$

$$ \sigma^{'}(z) = \sigma (z)\cdot (1-\sigma(z)) $$

```python
# STUDENT

def relu_backward(dA, Z):
    dZ = dA*np.where(Z>0,1,0)
    return dZ

def sigmoid_backward(dA, Z):
    # tip: make use of the "sigmoid"-function we implemented above 
    sig = sigmoid(Z)*(1-sigmoid(Z))
    dZ = dA * sig
    return dZ
```

## Task 6:

We now wish to implement a function `single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu")`, where:

* `dA_curr` corresponds to $\mathbf{dA}^{\{l\}}$, passed during back propagation, needed to calculate $\mathbf{dZ}^{\{l\}}$
* `W_curr` corresponds to $\mathbf{W}^{\{l\}}$, the current weight matrix
* `b_curr` corresponds to $\mathbf{b}^{\{l\}}$, the current bias vector 
* `A_prev` corresponds to $\mathbf{A}^{\{l-l\}}$, cached activation-values of the previous layer, needed to calculate $\mathbf{dW}^{\{l\}}$

`single_layer_backward_propagation` should calculate the gradients of the trainable parameters ($\mathbf{dW}^{\{l\}}, \mathbf{db}^{\{l\}}$) for a single layer $l$. It will also calculate $\mathbf{dZ}^{\{l\}}$ (depending on which activation function was used) in order to calculate $\mathbf{dA}^{\{l-1\}}$, which will be passed on to the preceding layer ${l-1}$ during back propagation. Use the formulae introduced above to perform the necessary calculations.

It returns `dA_prev`, `dW_curr`, `db_curr`, which correspond to $\mathbf{dA}^{\{l-1\}}$, $\mathbf{dW}^{\{l\}}, \mathbf{db}^{\{l\}}$

**IMPORTANT:** when calculating `db_curr`, make use of the function `np.sum(..., axis=..., keepdims=True)`, make sure you set `keepdims=True`, this will ensure that `db_curr` and `b_curr` keep the same dimensions, which is important in oder to perform parameter updates later.

```python
# STUDENT

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]
    
    # calculation of the activation function derivative
    dZ_curr = globals()[f"{activation}_backward"](dA_curr, Z_curr)
    
    # derivative of the matrix W
    dW_curr = dZ_curr.dot(A_prev.T)/m
    # derivative of the vector b
    db_curr = np.mean(dZ_curr, axis=1,keepdims=True)
    # derivative of the matrix A_prev
    dA_prev = W_curr.T.dot(dZ_curr)

    return dA_prev, dW_curr, db_curr
```

In the function `full_forward_propagation` you implemented above, you initialized the activations that feed in to the first layer with ${A^{\{0\}}} = X $, i.e. `A_curr = X`. We will also need values $\mathbf{dA}^{\{n\}}$ for or last layer $n$ in order to initialize back-propagation:

$$ \mathbf{dA}^{\{n\}} = \frac{\delta L}{\delta\mathbf{A}^{\{n\}}} = -(\frac{\mathbf{Y}}{\mathbf{\hat{Y}}} - \frac{1-\mathbf{Y}}{1-\mathbf{\hat{Y}}}) $$

where ${\hat{\mathbf{Y}}} = \mathbf{A}^{\{n\}}$ are our predicted values for the target variable. We have implemented this calculation for you:

```python
def loss_backward(Y, Y_hat):
    return - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
```

## Task 7:

Above we implemented the function `full_forward_propagation`, which sequentially iterates over the layers starting from the input layer in order to perfrom forward propagation. It calculates the transformations defined by the weight and bias parameters and activation functions, and stores the intermediate outputs in memory.

We now write a similar function called `full_back_propagation`, which iterates over the layers in reverse order, starting from the output layer in order to perform back propagation. It makes use of the intermediate outputs in order to calculate the gradients of the loss function with respect to the model parameters. It stores these gradients in a dictionary `grads_values`, which is returned in the end.

Complete the function `full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture)`, where:

* `Y_hat` corresponds to $\hat{\mathbf{Y}} = \mathbf{A}^{\{n\}}$
* `Y` corresponds to $\mathbf{Y}$, the training data labels, which are eihter 0 or 1
* `memory` is the dictionary of cached values for $\{\mathbf{Z}^{\{1\}}, \mathbf{Z}^{\{2\}}, ..., \mathbf{Z}^{\{n\}}\}$ and $\{\mathbf{A}^{\{1\}}, \mathbf{A}^{\{2\}}, ..., \mathbf{A}^{\{n\}}\}$
* `params_values` is the dictionary of current parameter values, i.e. $\{\mathbf{W}^{\{1\}}, \mathbf{W}^{\{2\}}, ..., \mathbf{W}^{\{n\}}\}$ and $\{\mathbf{b}^{\{1\}}, \mathbf{b}^{\{2\}}, ..., \mathbf{b}^{\{n\}}\}$
* `nn_architecture` is the dictionary that defines the model architecture

Here, you only have to call the single_layer_backward_propagation() function with the right parameters.

```python
# STUDENT

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = loss_backward(Y, Y_hat)
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(dA_curr,W_curr\
                                                                      ,b_curr,Z_curr,A_prev,activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values
```

Now that we have a way to get the gradients for all the trainable parameters, it is time to write a function that will allow us to update the parameters using the update rule introduced in the last exercise:

$$ \mathbf{W}^{\{l\}} = \mathbf{W}^{\{l\}} - \alpha \mathbf{dW}^{\{l\}} $$
$$ \mathbf{b}^{\{l\}} = \mathbf{b}^{\{l\}} - \alpha \mathbf{db}^{\{l\}} $$

## Task 8:

Complete the function `update` below. It takes the following parameters:

* `params_values` dictionary of parameter values $\{\mathbf{W}^{\{1\}}, \mathbf{W}^{\{2\}}, ..., \mathbf{W}^{\{n\}}\}$ and $\{\mathbf{b}^{\{1\}}, \mathbf{b}^{\{2\}}, ..., \mathbf{b}^{\{n\}}\}$
* `grads_values` dictionary of gradients for the trainable parameters $\{\mathbf{dW}^{\{1\}}, \mathbf{dW}^{\{2\}}, ..., \mathbf{dW}^{\{n\}}\}$ and $\{\mathbf{db}^{\{1\}}, \mathbf{db}^{\{2\}}, ..., \mathbf{db}^{\{n\}}\}$
* `nn_architecture` dictionary defining the neural network architecture
* `learning_rate`, the learning rate $\alpha$

```python
# STUDENT

def update(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]   
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;
```

## Task 9:
Now we have everything we need to train our model. The final task of this exercise, is to insert the functions you implemented above in the right places below. If you understood what you are doing, this should be more or less self-explanatory ;)

```python
# STUDENT
from IPython.display import clear_output
def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False):
    # initiation of neural net parameters
    
    params_values = init_layers(nn_architecture)
    
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []
    acc_test_history = []
    cost_test_history = []
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        Y_hat, cache = full_forward_propagation(X,params_values,nn_architecture)
        Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, NN_ARCHITECTURE)
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        
        cost_test = get_cost_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
        cost_test_history.append(cost_test)
        
        accuracy = get_accuracy_value(Y_hat, Y)

        acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
        accuracy_history.append(accuracy)
        acc_test_history.append(acc_test)
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat,Y,cache,params_values,nn_architecture)
        
        # updating model state
        params_values = update(params_values, grads_values,nn_architecture,learning_rate)
        
        if(i % 50 == 0):
            if(verbose):
                clear_output(wait=True)
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f} - cost-test: {:.5f} - accuracy-test: {:.5f}".format(
                    i, cost, accuracy, cost_test, acc_test))
                
            
    return params_values, cost_history, accuracy_history, acc_test_history, cost_test_history
```

```python
# Training
params_values, cost_history, accuracy_history, acc_test_history, cost_test_history = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), NN_ARCHITECTURE, 10000, 0.01, verbose=True)
```

```python
# Prediction
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, NN_ARCHITECTURE)
```

```python
# Accuracy achieved on the test set
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(acc_test))
```

And last but not least, let's plot how the accuracy and cost evolved over the training epochs...

```python
plt.style.use('fivethirtyeight')
plt.plot(np.arange(10000), np.array(cost_history))
plt.plot(np.array(cost_test_history))
plt.title("loss vs. epochs")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","test"])
```

```python
plt.plot(np.arange(10000), np.array(accuracy_history))
plt.plot(np.array(acc_test_history))
plt.title("accuracy vs. epochs")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","test"])
```

### Question 1:
What can you say about the learning progress of the model?


Both training and test accuracy are monotonouly increasing while cost or loss is monotonouly decreasing. (more or less)
We could also reduce the number of epochs, esp. with the reduced model, since accuracy is plateauing at about 6000 or 8000 epochs.


### Question 2:
Can you find out how many trainable parameters our model contains? Do you think that this number of parameters is appropriate for our classification task?


You can achieve similar results with just 201 or even 101 parameters. Just commenting out every hidden layer, still yields acceptable results. Using over 5000 parameters to find to classify two rings of data seems like a lot

```python
np.sum([x.size for x in params_values.values()])
```

Congratulations, you made it through the sixth tutorial of this course!

# Submitting your assignment

Please rename your notebook under your full name and **submit it on the moodle platform**. If you have problems to do so, you can also send it again to machinelearning.dhc@gmail.com

Please rename the file to 1_LinRegTut_<GROUP\>.ipynb and replace <GROUP\> with your group-name.

As this is also the first time for us preparing this tutorial, you are welcome to give us feedback to help us improve this tutorial.  

Thank you!  

Jana & Remo

```python

```
