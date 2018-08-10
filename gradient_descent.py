"""
This script implements the gradient descent algorithm that is used
to optimize functions and find their local minima. In this example,
the algorithm is demonstrated on linear regression.
"""
import matplotlib.pyplot as plt
import numpy as numpy
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def data_generation():
    """
    This function downloads the data and standardizes it
    using built in functions from sklearn.
    """ 
    hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0)
    hitters=hitters.dropna()

    X=hitters.drop('Salary',axis=1)
    X=pd.get_dummies(X,drop_first=True)
    y=hitters.Salary

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler = preprocessing.StandardScaler().fit(y_train.values.reshape(-1, 1))
    y_train = scaler.transform(y_train.values.reshape(-1, 1)).reshape((-1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1)).reshape((-1))


    X_train = preprocessing.add_dummy_feature(X_train)
    X_test = preprocessing.add_dummy_feature(X_test)

    return X_train, X_test, y_train, y_test

def objective_function(X, y, beta):
    """
    This function implements the objective function for linear 
    regression. Beta contains the coefficients.
    """
    return np.subtract(y, np.dot(X, beta))
    

def gradient_of_function(X, y, beta):
    """
    This function computes the gradient of the linear regression
    function that we implemented above.
    """
    return np.dot(X.T, (np.subtract(y, np.dot(X, beta)))) 

def graddescent(t,max_iter,X,y):
    """
    This function implements the gradient descent algorithm. It 
    uses the objective_function() and gradient_of_function() 
    functions which implement the objective function and its 
    gradient respectfully.

    Args:
        max_iter(int): The maximum number of iterations that the algorithm 
                       should run for.
        X(array): The predictor variables, in the form of an array
        y(array): The target variables
    
    Returns:
        beta(array): The values of beta that yield the lowest objective function value
        obj_vals(array): The value of the objective function at each iteration

    """
    beta = np.zeros(X.shape[1])
    grad_b = gradient_of_function(X, y, beta)
    iter = 0
    obj_vals = []
    while iter < max_iter:
        beta = np.subtract(beta,t*grad_b)
        obj_vals.append(objective_function(X, y, beta))
        grad_b = gradient_of_function(X, y, beta)
        iter += 1
    return np.array(beta), obj_vals

def main():
    X_train, X_test, y_train, y_test = data_generation()
    betas, objs = graddescent(0.1,100, X_train, y_train)
    plt.plot(objs)
    plt.title("Objective function value by iteration")
    plt.xlabel("iterations (t)")
    plt.ylabel(r'$F(\beta)$')

if __name__ == '__main__':
    main()    