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

# Tutorial 5.1: Logistic regression and Gaussian Processes

Welcome to the fifth tutorial of the course 'Machine learning for Precision Medicine'.

In this exercise, we want to predict benign and malignant breast cancer from biopsy images with a logistic regression model. We herefore use the WDBC dataset containing 
- 569 samples from patients with known diagnosis
- 357 benign
- 212 malignant
- 30 features extracted from fine needle aspirate slides

We are given a number of features that describe the cell nuclei that have been determined from image processing techniques [Street et al, 1992].

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import util
```

For this exercise, we have outsourced some functions, which we frequently use in a second python script 'util.py'. 

```python
#load the data with the load_data() function in util.py
X_train, y_train = util.load_data('data.csv')
X_test, y_test = util.load_data('data.csv', testing_data=True)
```

```python
X_train.head()
```

```python
y_train[0:20] # 1 where malignant, 0 otherwise
```

<!-- #region -->
### Binary Classification

**Classification** refers to the task of predicting a **class label** $y$, *i.e.*, the diagnosis, from a **feature vector** $\bf{x}$.
For the case, where $y$ can take one of two values, we speak of binary classification.


For the task at hand, this means that for the image features, we are
then given a new image for which we don't know the diagnosis. We can predict the diagnosis based on what we have learned from from the training data.

Similar to linear regression, we are trying to find an optimal weight vector $\mathbf{w}$ that will minize our objective function. However, we have to make adjustments because we are no longer dealing with continous output variables, but binary class labels. In most real-world datasets, the two classes will not be perfectly separable, i.e. we expect to make mistakes and want to minimize those. Therefore we predict probabilities of belonging to class 1, instead of binary class labels. We can achieve this by making use of the logistic sigmoid link function. 

The **logistic** $\pi(a)$ is a function between 0 and 1, making it suited for modeling probabilities. It is called a **sigmoid** function because of its *s*-shape.

\begin{equation}
\pi(a) := \frac{1}{1+\exp{\left(-a\right)}} = \frac{\exp{\left(a\right)}}{1+\exp{\left(a\right)}}
\end{equation}

Here, as we are modeling linear functions, $a=\mathbf{x}_n\mathbf{w}$, where $\mathbf{x}_n$ is the **feature vector** for the $n$-th individual (given), and $\mathbf{w}$ is a **weight vector** that we would like to find.

<!-- #endregion -->

## Task 1: ##  
Implement the logistic sigmoid following the formula above. We have to make sure that the output of this function is never exactly 0 or 1, because we will take the log of it. Therefore, we use np.clip() to limit range (from ... to).

```python
def logistic(a):
    logist = # your_code
    logist = # your_code
    return logist
```

As in every machine learning problem, we need to define an **objective** function $L$. It quantifies the error between our predictions and the ground-truth. We then determine the weights $\mathbf{w}^{opt}$ that minimize $L$.

We would like to assign high probability to all the instances that belong to the target class and low probability otherwise. Accordingly, we would like to obtain a function that records a loss, whenever we assign low probability to the correct class $c_{true}$.

One function that achieves this is the **log-loss** or **cross-entropy** loss, which is defined as:

\begin{equation}
loss = -\sum_{n\in c_1} \ln( \pi(\mathbf{x}_n\mathbf{w}) ) - \sum_{n'\in c_2} \ln( 1-\pi(\mathbf{x}_{n'}\mathbf{w}) )
\end{equation}



Where $c_1$ are all the members of the first class (here: malignant, `y == 1.`) and $c_2$ are all the members of the second class (benign, `y == 0.`).


## Task 2:
Implement the log-loss (binary cross entropy function) using the formula above. 

```python
def logloss(y, y_hat):
    """
    return the loss for predicted probabilities y_hat, and class labels y
    Keyworld arguments
    y -- scalar or numpy array
    y_hat -- scalar or numpy array
    """

    loss = #your_code
    
    return loss
    
```

```python
logloss(np.array([0.,1.,1.]), np.array([0.1, 0.5, 0.99]))
```

** Expected output **:
0.80855803207127308


Taken together, we obtain the **Logistic Regression objective** $L(\mathbf{w})$.
\begin{equation}
L(\mathbf{w}) = \underbrace{-\sum_{n\in c_1} \ln( \pi(\mathbf{x}_n\mathbf{w}) ) - \sum_{n'\in c_2} \ln( 1-\pi(\mathbf{x}_{n'}\mathbf{w}) )}_{loss} + \underbrace{ \lambda \cdot 0.5 \cdot \sum_{d=1}^{D}{w_d}^2}_{regularizer}
\end{equation}


## Task 3:  
Implement the regularizer

```python
def regularizer(w, lambd):
    '''
    return the value for the regularizer for a given w and lamd
    ''' 
    reg = # your_code
    return reg
```

### The derivative

In order to minimize our objective function, we will make use of the derivative. The derivative will tell us in which direction we have to adjust our weights, in order to minimize the loss. Note that no analytical solition exists. Instead, we will have to optimize our objective using an optimization algorithm (see below). The derivative of the objective with respect to a single $w_d$ is defined as follows:

\begin{equation}
\frac{\partial L}{\partial w_d} = \sum_n^{N}{x_{nd}} \cdot
 \left( \pi\left(\mathbf{x}_n\mathbf{w}_n\right)-I\left(\mathbf{y}_n== c_1\right)\right) + \lambda \cdot w_d
\end{equation}

With $I$ being the Identity matrix. $I(a==b)$ denotes the indicator function, which yields 1 if $a=b$ and 0 otherwise.

The sign of the derivative indicates the direction in which the objective gets larger or smaller and the magnitude the rate.


### The gradient

By stacking all partial derivatives into a single vector, we obtain the gradient $\nabla_\mathbf{w} (L)$.

\begin{equation}
\nabla{L}\left(\mathbf{w}^{t}\right) =
\left[\begin{matrix}
\frac{\partial L}{\partial w^t_1}\\
\vdots\\
\frac{\partial L}{\partial w^t_D}
\end{matrix}\right]
=
\underbrace{\mathbf{X}^{T}
 \left( \pi\left(\mathbf{X}\mathbf{w}^t\right)-I\left(\mathbf{y}==c_1\right)\right)}_{\nabla{\text{loss}}\left(\mathbf{w}^{t}\right)}+ \underbrace{\lambda \cdot \mathbf{w}^t}_{\nabla{\text{regularizer}}\left(\mathbf{w}^{t}\right)}
\end{equation}

$\nabla_\mathbf{w} (L)$ is a $D$-dimensional vector pointing in the direction of steepest growth of the objective and in the opposite direction in which the steepest reduction.
Using the gradient, we can define a simple optimization algorithm.

#### Steepest descent

The steepest descent algorithm uses the gradient by making small steps in the direction $-\nabla_{\mathbf{w}^{t}} (L)$. You can think about it as being on a hill and descending the hill in the steepest direction downwards.
Therefore the algorithm is called **steepest descent**.

given learning rate $0<\alpha<1.0$ and current weight estimate $\mathbf{w}^{t}$.
Iterate by setting $\mathbf{w}^{t+1} = \mathbf{w}^{t} - \alpha \cdot \nabla_{\mathbf{w}^{t}} (L)$.

A typical value for $\alpha$ is around $10^{-4}$.

A problem with steepest descent is that the estimate tends to oscillate and often even overshoots and diverges (leading to an increase in the objective). Getting the learning rate right is very hard, trading off progress in learning and risk of diverging. Many tricks exit to improve learning in gradient descent, such as weight decay, where the learning rate is gradually reduced during learning.

Here we will implement the steepest_descent_optimizer as a class, which has the attributes: alpha, lamd, X, y, w and max_iter (maximum number of iterations) and the functions: predict(), optimize()


## Task 4:

We will now implement the algorithm described above using a class Steepest_descent_optimizer(). 

You are given the template below, and are expected to complete the methods `_gradient()`, `_update()` and `optimize()`.

Remember within the class you can access the current weights, X, y, value for the regularizer (`lambd`), etc using `self.w`, `self.X`, `self.y` and so on.

The `_gradient()` method should use `self.X`, `self.y`, `self.w`, `self.lambd` ($\lambda$) the `logistic()` function you implemented above, and return the gradient of the loss function with respect to `self.w` ($\mathbf{w}$) (see formula above for $\nabla{L}\left(\mathbf{w}^{t}\right)$). Tip: for $(y == c_1)$ you can just use `self.y` directly.

The `_update()` method should get the gradient using `self._gradient()` and update the current weights `self.w` using the learning rate (`self.alpha`) and the update rule for steepest descent described above: $\mathbf{w}^{t+1} = \mathbf{w}^{t} - \alpha \cdot \nabla_{\mathbf{w}^{t}} (L)$

Finally, the `optimize()` function will update the weights for `self.max_iter` times, and after every update calculate the loss (including the regularizer) and store it in a list `loss`. Finally, it returns this `loss`-"history".


```python
class Steepest_descent_optimizer():
    
    def __init__(self,X,y,lambd,alpha):
        self.alpha = alpha
        self.lambd = lambd
        
        self.X = X
        self.y = y
        
        self.w = np.zeros(X.shape[1]) # we initialize the weights with zeros
        
        self.max_iter = 10000 # set the max number of iterations
    
    def _gradient(self):
        # calculate the gradient of w 
        # your_code
        return grad
    
    def _update(self):
        grad = self._gradient()
        # update the weights using the gradient and learning rate
        self.w =  # your_code
        
    def predict(self, X):
        return logistic(X.dot(self.w))
        
    def optimize(self):
        it = 0
        loss = []
        # we iterate until we reach self.max_iter
        while it < self.max_iter:
            # update the weights (use the method you implemented above)
            # append the current loss (use self.predict, and the regularizer(), and logloss() functions)
            
            # your_code
            
            loss.append(# your_code
            it += 1
        return loss
```

```python
# Create an instance of the class
optimizer = Steepest_descent_optimizer(X_train, y_train, lambd = 0.001, alpha = 0.001)

# run the optimization for 10000 steps, this might take a while...
loss = optimizer.optimize()
```

```python
optimizer.w
```

```python
# Plot  the evolution of the loss
import matplotlib.pyplot as plt
plt.plot(np.arange(len(loss)),np.array(loss))

```

```python
# Predict from the test set
test_pred = optimizer.predict(X_test)
```

```python
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
```

```python
import importlib
importlib.reload(util)
```

Let's see how accurate our predictions are using the average precision score and the roc area under the curve score.

```python
average_precision_score(y_test, test_pred)
```

** Expected output **:  ~ 0.993

```python
roc_auc_score(y_test, test_pred)
```

** Expected output **: ~ 0.995

```python
util.plot_confusion_matrix(test_pred, y_test)
```

Congratulations, you made it through one part of the fifth tutorial of this course!

# Submitting your assignment

Please rename your notebook under your full name and **submit it on the moodle platform**. If you have problems to do so, you can also send it again to machinelearning.dhc@gmail.com

Please rename the file to 1_LinRegTut_<GROUP\>.ipynb and replace <GROUP\> with your group-name.

As this is also the first time for us preparing this tutorial, you are welcome to give us feedback to help us improve this tutorial.  

Thank you!  

Jana & Remo

```python

```
