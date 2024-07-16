# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:20:09 2024

@author: oscar
"""

from pathlib import Path
import numpy as np
from mlp_classifier import MLPClassifier


def get_data():
    '''Data originally from: http://yann.lecun.com/exdb/mnist/
    Consists of labeled images of handwritten digits. The training set contains 
    60k images, while the testing set contains 10k images. Each image is in 
    grayscale and is 28x28 pixels. These are converted into vectors of length
    28x28=784, with entries in the interval [0,1]. 
    '''
    
    data = dict()
    for f in Path(Path.cwd(), 'data').glob('*'):
        with open(f, 'rb') as file:
            data[f.stem] = np.frombuffer(file.read(), dtype=np.uint8)
    
    data['train-images'] = data['train-images'][16:].reshape((-1,28**2)) / 255
    data['train-labels'] = data['train-labels'][8:]
    
    data['test-images'] = data['test-images'][16:].reshape((-1,28**2)) / 255
    data['test-labels'] = data['test-labels'][8:]
    
    return data



### Example ###
data = get_data()    
nn = MLPClassifier([784, 89, 10])

nn.fit(data['train-images'], data['train-labels'])
dist = np.array([nn.predict(x) for x in data['test-images']])
preds = np.argmax(dist, axis=1)
accuracy = (preds == data['test-labels']).sum() / preds.size 
    

# Accuracy of 97.71% was achieved on the MNIST test set (10k examples) 
# with a 3-layer MLP with 89 hidden neurons. Weights and biases were
# initialized uniformly at random from the interval [-0.05, 0.05]. 
# The training process lasted for 20k steps at a learning rate of 0.1, 
# where stochastic gradient descent was employed with 5% out of the 60k
# training examples. 
