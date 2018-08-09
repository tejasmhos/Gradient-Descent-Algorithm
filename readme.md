# Gradient Descent Algorithm

## Introduction

Gradient descent is an iterative optimization algorithm that is used to find the minimum of a function. To find a local minimum of any given function, we need to take steps that are proportional to the negative of the gradient of the function at the current point. Eventually, we will reach a local minimum.

## Prerequisite Conditions to Apply the Algorithm

The gradient descent algorithm can be applied to a given function, provided that it's:

- Differentiable
- Convex 

### Differentiability

A function is differentiable if a derivative exists at each point in its domain. 

![](/images/Polynomialdeg3.png)

The graph above shows a function that is differentiable. An example of a non-differentiable function is the absolute value function. It is differentiable at all points, except for the point at which the value of the variable is 0.

### Convex

A function is convex if its epigraph is a convex set. Basically, the region above the graph should be a convex set. 



The gradient descent algorithm can be used only if these two conditions are met. 

## Algorithm

The gradient descent algorithm is reproduced below:

![](/images/algorithm.png)

The input given is a step size. We also define a stopping criterion. In this implementation, we just use a maximum iteration value as the stopping criterion. Once the stopping criterion is met, the value that we obtain is the optimized value of the objective function.

This code runs gradient descent to optimize the value of the coefficients for a linear regression based model. All the code is contained in the gradient_descent.py file. 
