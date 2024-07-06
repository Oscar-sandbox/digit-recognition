# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:47:30 2024

@author: oscar
"""
import itertools 
from pathlib import Path

import numpy as np


def get_data():
    '''data from: http://yann.lecun.com/exdb/mnist/'''
    
    data = dict()
    for f in Path(Path.cwd(), 'data').glob('*'):
        with open(f, 'rb') as file:
            data[f.stem] = np.frombuffer(file.read(), dtype=np.uint8)
    
    data['train-images'] = data['train-images'][16:].reshape((-1,28**2)) / 255
    data['train-labels'] = data['train-labels'][8:]
    
    data['test-images'] = data['test-images'][16:].reshape((-1,28**2)) / 255
    data['test-labels'] = data['test-labels'][8:]
    
    return data


def sigmoid(x): 
    return  1/(1 + np.exp(-x))


def sigmoid_diff(x): 
    return np.exp(-x)/(1 + np.exp(-x))**2


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.depth = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.W, self.b = self.initialize_parameters(mode='random')
        self.train_stats = dict()
    
    
    def initialize_parameters(self, mode='zeros'):
        W, b = [None], [None]
        if mode == 'zeros':
            for m, n in itertools.pairwise(self.layer_sizes):
                W.append(np.zeros((n, m)))
                b.append(np.zeros(n))
        elif mode == 'random':
            for m, n in itertools.pairwise(self.layer_sizes):
                W.append(2*np.random.random_sample((n, m)) - 1)
                b.append(2*np.random.random_sample(n) - 1)
        return W, b

            
    def backprop(self, layer, y):
        dW, db = self.initialize_parameters()
        z = [None]   # z^l = W^l @ a^{l-1} + b^l
        a = [layer]  # a^l = sigmoid(z^l) 
        
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
    
    
    def train(self, features, labels, steps=100, alpha=1, sgd_percent=0.01):
        self.train_stats['steps'] = steps
        self.train_stats['alpha'] = alpha
        self.train_stats['sgd_percent'] = sgd_percent
        
        rng = np.random.default_rng(seed=42)
        population_size = len(features)
        sample_size = int(sgd_percent * population_size)
        
        for step in range(steps):
            dW, db = self.initialize_parameters()
            
            for _ in range(sample_size):
                i = rng.integers(population_size)
                feature, label = features[i], labels[i]  
                y = np.zeros(self.layer_sizes[-1])
                y[label] = 1.0                
                
                dW0, db0 = self.backprop(feature, y)
                for l in range(1, self.depth):
                    dW[l] += dW0[l]
                    db[l] += db0[l] 
            
            for l in range(1, self.depth):
                self.W[l] -= alpha * dW[l]/sample_size
                self.b[l] -= alpha * db[l]/sample_size
                
            
    def predict(self, layer):
        for l in range(1, self.depth):
            layer = sigmoid((self.W[l] @ layer) + self.b[l])
        return np.exp(layer) / np.exp(layer).sum()
      

