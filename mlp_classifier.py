# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:47:30 2024
@author: oscar

This module implements a Multilayer Perceptron from scratch to classify 
handwritten digits, a classical example in the field of Deep Learning. 
Training examples for this task can be found in the famous MNIST database, 
available originally at http://yann.lecun.com/exdb/mnist/. 

The classifier obtains a 97.71% accuracy on the MNIST test set, which is 
in line with the results of other classifiers employing a similar strategy.
"""

import itertools 
import numpy as np


def sigmoid(x): 
    '''Neuron activation function. Here, ReLU is used.'''
    return np.maximum(x, 0)


def sigmoid_diff(x): 
    '''Derivative of the neuron activation function.'''
    return np.heaviside(x, 0)


class MLPClassifier:
    def __init__(self, layer_sizes):
        '''Creates a Multilayer Perceptron Classifier.  

        Parameters
        ----------
        layer_sizes : list[int]
            Indicates the size of the MLP, where len(layer_sizes) is the depth
            of the MLP and layer_sizes[i] is the number of neurons in the 
            ith layer.

        Returns
        -------
        None.
        '''
        
        self.depth = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.W, self.b = self._initialize_parameters(mode='random')
        
        self.train_stats = dict()
        self.rng = np.random.default_rng(seed=42)
    
    
    def _initialize_parameters(self, mode='zeros'):
        '''Initializes the weights and biases of the MLP. 

        Parameters
        ----------
        mode : str, optional
            If mode is "zeros", then every weight matrix and bias vector is set
            to zero. Otherwise, if mode is "random", weights and biases are 
            uniformly picked from the interval [-0.05, 0.05]. The default is 
            "zeros".

        Returns
        -------
        W : list[numpy.ndarray]
            List of weight matrices, where W[i] is the weight matrix between
            the (i-1)th and ith layers. W[0] is set to None.
        b : list[numpy.ndarray]
            List of bias vectors, where b[i] is the bias vector of the ith 
            layer. b[0] is set to None.
        '''
        
        W, b = [None], [None]
        if mode == 'zeros':
            for m, n in itertools.pairwise(self.layer_sizes):
                W.append(np.zeros((n, m)))
                b.append(np.zeros(n))
        elif mode == 'random':
            for m, n in itertools.pairwise(self.layer_sizes):
                W.append(0.1*np.random.random_sample((n, m)) - 0.05) #!
                b.append(0.1*np.random.random_sample(n) - 0.05)
        return W, b

            
    def _backprop(self, x, y):
        '''Calculates the gradient of the cost function for a single training
        example.

        Parameters
        ----------
        x : numpy.ndarray
            Training example to be fed to the MLP Classifier. len(x) must be 
            equal to the number of neurons on the first layer. 
        y : numpy.ndarray
            Expected output of the MLP Classifier. len(y) must be equal to the 
            number of neurons on the last layer.

        Returns
        -------
        dW : list[numpy.ndarray]
            List of weight matrices, where dW[i] represents the derivative of 
            the cost function with respect to the weights in self.W[i].
        db : list[numpy.ndarray]
            List of bias vectors, where db[i] represents the derivative of 
            the cost function with respect to the biases in self.b[i].
        '''
        
        dW, db = self._initialize_parameters()
        z = [None] # z^l = W^l @ a^{l-1} + b^l (input)
        a = [x]    # a^l = sigmoid(z^l)        (output)
        
        for l in range(1, self.depth):
            z.append(self.W[l] @ a[l-1] + self.b[l])
            a.append(sigmoid(z[-1]))
        
        z_diff = (a[-1] - y) * sigmoid_diff(z[-1])  # dC/dz^l 
        for l in range(self.depth-1, 0, -1):
            if l < self.depth-1:
                z_diff = (self.W[l+1].T @ z_diff) * sigmoid_diff(z[l]) 
            dW[l] = np.outer(z_diff, a[l-1])
            db[l] = z_diff
        
        return dW, db
    
    
    def fit(self, X, y, steps=100, alpha=0.1, sgd_percent=0.05):
        '''Fits the MLP Classifier to the training data. 
        
        Parameters
        ----------
        X : list[numpy.ndarray]
            List of training examples to be fed to the MLP classifier. For 
            every x in X, len(x) must be equal to the number of neurons in 
            the first layer. 
        y : list[int]
            List of labels of the training examples.
        steps : int, optional
            Number of steps taken in the training process. The default is 100.
        alpha : float, optional
            Learning rate, where 0 < alpha < 1. Lower values of alpha yield 
            more precise results, but require a higher number of steps. The 
            default is 0.1.
        sgd_percent : float, optional
            Percent of the training examples used during stochastic gradient 
            descent. Higher values of sgd_percent yield more precise results, 
            but each step requires more time to compute. The default is 0.05.

        Returns
        -------
        None.
        '''
        
        self.train_stats['steps'] = steps
        self.train_stats['alpha'] = alpha
        self.train_stats['sgd_percent'] = sgd_percent
        sample_size = int(sgd_percent * len(X))
        
        for step in range(steps):
            dW, db = self._initialize_parameters()
            
            for _ in range(sample_size):
                i = self.rng.integers(len(X))
                y_layer = np.zeros(self.layer_sizes[-1])
                y_layer[y[i]] = 1
                
                dW0, db0 = self._backprop(X[i], y_layer)
                for l in range(1, self.depth):
                    dW[l] += dW0[l]
                    db[l] += db0[l] 
            
            for l in range(1, self.depth):
                self.W[l] -= alpha * dW[l]/sample_size
                self.b[l] -= alpha * db[l]/sample_size
                
            
    def predict(self, x):
        '''Predicts the label of a test example.
        
        Parameters
        ----------
        x : numpy.ndarray
            Testing example to be fed to the MLP Classifier. len(x) must be 
            equal to the number of neurons on the first layer. 

        Returns
        -------
        dist : numpy.ndarray
            Probability distribution of the classification, where dist[i] 
            equals the probability that i is the label of x. The final
            decision of the classifier is argmax(dist). 
        '''
        
        for l in range(1, self.depth):
            x = sigmoid((self.W[l] @ x) + self.b[l])
        dist = np.exp(x) / np.exp(x).sum()
        return dist
