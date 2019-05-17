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

# Tutorial 4: Bayesian Linear Regression and Gaussian Processes

Welcome to the fourth tutorial of the course 'Machine learning for Precision Medicine'.
 
In the last exercise we predicted the insurance cost of a customer, based on 6 health variables using basis functions and kernelized regression. Now we want to add uncertainty estimates to our predictions (i.e. standard deviations of the expected cost). 

As we will see this requires only minimal adjustments to our original kernelized Ridge regression model. For this reason this week's tutorial will be a bit shorter than usual.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# we will make use of the sklearn.metrics.pairwise_kernels function:
from sklearn.metrics.pairwise import pairwise_kernels
```

We will load the data and perform the same pre-processing as before, i.e. one-hot encoding and standardization.

```python
# load the data, and check out the first couple of rows
data = pd.read_csv('./insurance.csv', header=0)
data_orig = data
```

```python
# 1-hot encoding for the categorical variables
data = pd.get_dummies(data, columns=['region','smoker','sex'])
print('old variables: {}'.format(list(data_orig.columns)))
# we get rid of the redundant columns:
data.drop(labels=['smoker_no','sex_male','region_northeast'], axis=1, inplace=True)
print('new variables: {}'.format(list(data.columns)))
```

```python
# Set dependent and independent variables 
dep_var = ['charges']
indep_var = ['age','bmi','children','sex_female','smoker_yes','region_northwest','region_southeast','region_southwest']

# standardize the data 
def zscore(x):
    return np.array((x - np.mean(x)) / (np.std(x)))

data = data.apply(zscore)

# Plot the distributions and see how the data is associated with cost (charges)
fig, ax = plt.subplots(2,len(indep_var),figsize=(24, 5))
for i, v in enumerate(indep_var):
    ax[0,i].hist(data[v])
    ax[0,i].set_xlabel('')
for i, v in enumerate(indep_var):
    ax[1,i].scatter(data[v],data[dep_var].values, alpha=0.2)
    ax[1,i].set_xlabel(v)
```

```python
# define the training set
X = data[indep_var].values
y = data[dep_var].values

# split in to train, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2)
```

## Introduction

In our previous exercise on kernelized linear regression, we made use of the following formula to estimate the dual variable $\mathbf a$ (our regression parameters), which, taking into account the L2 regularization penalty weighted by $\lambda$, minimizes the mean squared error of our kernelized regression: 

 $$ \mathbf a = (\mathbf K + \lambda \mathbf I)^{-1}\mathbf y$$

Where $\mathbf{K}$ is the kernel matrix of our training observations $\mathbf{X}$.

We obtained this result by setting the gradient of the objective function ($E_2$) below to zero with respect to $\mathbf{a}$.

$$ E_2(\mathbf{a}) = 0.5*\mathbf{a}^T \mathbf{K}\mathbf{K} \mathbf{a} - \mathbf{a}^T\mathbf{K}\mathbf{y} + 0.5 * \mathbf{y}^T\mathbf{y} + 0.5*\lambda \mathbf{a}^T\mathbf{K}\mathbf{a}^T$$

where $\mathbf{y}$ is our response variable (see lecture slide 15, lecture 4 - *Basis Functions and Kernels*). The additive term on the right denotes the regularization penalty, while the others relate to the mean squared error. We then used cross-validation to find an optimal value for $\lambda$.



In the lecture *Bayesian Linear Regression and Gaussian Pocesses* we discussed the Bayesian interpretation of linear regression. In a Bayesian setting, we no longer predict simple point-estimates for our outcome variables ($\mathbf{y}^*$), but rather our predictions themselves become distributions, conditioned on the training data ($\mathbf{X}, \mathbf{y}$), the new observations ($\mathbf{X}^*$), our choice of kernel $\theta_K$ (implying a choice of basis function) and our prior (which we will parameterize by $\sigma^2$).

The uncertainty in our predictions then factors into two parts. One part comes from noise, which is inherent in the data (and is irreducible!). The second part arises from uncertainty in our model parameters. This second part is due to only ever observing a finite amount of training data and decreases as the amount of training data increases.

We then found that this predicitve distribution, which we will call $\mathbf{f}^*$ is itself a Gaussian distribution, if our model weights and the noise follow Gaussian distributions and the predictions are linear kombinations of (the transformed) input variables. Is is what is referred to as a Gaussian process.

In fact, we found out that adding an L2-penalty to our model weights is equivalent to placing a prior over our model weights assuming mean $\mu = 0$ and variance parameterized by $\lambda$.



Finally, because the predictive distribution $\mathbf{f}^*$ is a multivariate Gaussian, it can be described by a vector of mean values $\mu^*$ and a covariance matrix ${\Sigma^*}$. I.e:


$$p(\mathbf{f}^{*}|\mathbf{X},\mathbf{y},\mathbf{X}^*,\theta_K,\sigma^2)=\mathcal{N}(\mu^*, \Sigma^*)$$

For reference, please consider lecture slide 34.

Now we want to calculate the mean and covariance matrix of our predictive distribution by completing the square (tip: look up the mathematical term "completing the square" if you do not know what this means). The mean value of our predictive distribution is based on the Kernel matrix ($\mathbf{K_{X^*,X}}$) between our new observations ($\mathbf{X}^*$) and our training observations ($\mathbf{X}$), $\mathbf{y}$ and ${\sigma^2}$:

$$\mu^* = \mathbf{K_{X^*,X}}[\mathbf{K_{X,X}}+\sigma^2 \mathbf I]^{-1} \mathbf y$$

**Question 1**:

See how the formula above relates to how we made predictions with our kernelized ridge regression model. What is the interpretation of $\sigma^2$?

```
the variance of our model weights, used to be lambda!
```

The covariance matrix ${\Sigma^*}$ is calculated the following way:

$$\Sigma^* =  \mathbf{K_{X^*,X^*}} - \mathbf{K_{X^*,X}}[\mathbf{K_{X,X}}+\sigma^2 \mathbf I]^{-1}\; \mathbf{K_{X,X^*}}$$

Using the two formulae above we will calculate the mean and covariance matrix of our predictive distribution. We will use the value for $\lambda$ which we found in the last exercise as our $\sigma^2$ for the rbf-kernel.


## Task 1: ##  
Complete the function `predictive_Gauss` below. It takes the following parameters:

- `X`: training data X (matrix, `np.ndarray`)
- `y`: training data y (vector, `np.ndarray`)
- `X_star`: new observations X (matrix, `np.ndarray`)
- `metric`: metric passed to the `pairwise_kernels()` function (see imports in the beginning of the exercise, `str`)
- `sigma_sq`: sigma-squared parameter (`float`)

It returns `mu_star` and `sigma_star` defined above.


$$\Sigma^* =  \mathbf{K_{X^*,X^*}} - \mathbf{K_{X^*,X}}[\mathbf{K_{X,X}}+\sigma^2 \mathbf I]^{-1}\; \mathbf{K_{X,X^*}}$$

$$\mu^* = \mathbf{K_{X^*,X}}[\mathbf{K_{X,X}}+\sigma^2 \mathbf I]^{-1} \mathbf y$$

```python
def predictive_Gauss(X,y,X_star, metric, sigma_sq):
    
    assert type(X) == np.ndarray
    assert type(y) == np.ndarray
    assert type(X_star) == np.ndarray
    assert type(sigma_sq) == float
    
    #training data corresponds in size
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == X_star.shape[1]

    K = pairwise_kernels(X, metric=metric)
    K_xstar_x = pairwise_kernels(X_star, X, metric=metric)
    K_xstar_xstar = pairwise_kernels(X_star, X_star, metric=metric)
    K_x_xstar = pairwise_kernels(X, X_star, metric=metric)
    
    #calculate mu_star: mean of predictive distribution
    mu_star = K_xstar_x.dot(np.linalg.inv(K + sigma_sq*np.eye(len(K)))).dot(y)

    #calculate sigma_star: covariance matrix
    sigma_star = K_xstar_xstar - K_xstar_x.dot(np.linalg.inv(K + sigma_sq * np.eye(len(K)))).dot(K_x_xstar)

#     print(f'shape of sigma star: {sigma_star.shape}')
#     print(f'shape of X: {X.shape}')
#     print(f'shape of y: {y.shape}')
#     print(f'shape of X_star: {X_star.shape}')
    
    return mu_star, sigma_star
```

**Question: **  
What is the shape of $\Sigma^*$ if you predict for $N$ new observations? How can you retrieve the variance of a single prediction $y^*_i$ from $\Sigma^*$?


Shape of $\Sigma^*$: (268, 268), which is $N$ * $N$, where $N$ is a number of new observations

To retrieve variance of a single prediction $y^*_i$ from $\Sigma^*$:  

Retrieve values from the diagonal of the covariance matrix sigma_star, where the covariance of a single y with a single y equals the variance of that y.

for example: variance of y_1: `sigma_star[1,1]`

```python
best_lambd = 0.15199110829529347
mu_star, sigma_star = predictive_Gauss(X_train,y_train,X_test, 'rbf', best_lambd)

print('\nmu_star[10]: {}\nsigma_star[10,10]: {}'.format(mu_star[10],sigma_star[10,10]))
print('No expected output this time! ;)')
```

```python
print(sigma_star)
```

## Task 2: ## 
Retrieve the standard deviation for each all predictions $\mathbf{y}^*$ (`sigma_star`):

```python
standard_deviations = np.sqrt(np.diagonal(sigma_star))
```

```python
# we plot our predictions with error-bars denoting +-2*(standard deviation) 

plt.figure(figsize=(12, 12))

plt.errorbar(y_test, mu_star, 2*np.sqrt(np.diagonal(sigma_star)), fmt='none', alpha=0.5)
plt.scatter(y_test, mu_star, s=9, c='r', label='mu_star')
plt.scatter(y_test, y_test, s=9, c='g', label='y_star')
plt.legend()

# you should now see a scatter-plot with red points, blue-error bars and green points on the diagonal:
```

Congratulations, you made it through the fourth tutorial of this course!  

# Submitting your assignment

Please rename your notebook under your full name and **submit it on the moodle platform**.

Please rename the file to 4_BayesReg_<GROUP\>.ipynb and replace <GROUP\> with your group-name.

As this is also the first time for us preparing this tutorial, you are welcome to give us feedback to help us improve this tutorial.  

Thank you!  

Jana & Remo

```python

```
