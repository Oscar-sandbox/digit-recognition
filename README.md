![ ](data/portada.jpg)

# Digit Recognition with Deep Learning

This project employs the techniques of Deep Learning to tackle a classic task 
in Artificial Intelligence: Optical Character Recognition (OCR). Specifically, 
this project aims to construct a classifier for handwritten digits. 

The code implements an artificial neural network from scratch in Python, using 
Numpy. In more detail, the project provides an object for a Multilayer 
Perceptron Classifier, with common methods such as `fit` and `predict`. 
Although many libraries already offer classifiers of this type, there is value
in understaing the mathematics underlying these structures, as this allows for 
greater control of the networks' parameters and their learning process. 

# About the Training Set

As any other classifier in Deep Learning, we need a vast database of existing 
examples to feed to the neural network. Luckily, training examples for this 
task can be found in the famous MNIST database, available [here](http://yann.lecun.com/exdb/mnist/). The 
database contains a training set with 60k images, and a testing set with 10k
images. Each image is in grayscale, is 28x28 pixels in size and is labeled with 
its corresponding digits from 0 to 9. 

# Results

An acuracy of 97.71% was achieved on the MNIST test set (10k examples) 
with a 3-layer MLP Classifier with 89 hidden neurons. Weights and biases were
initialized uniformly at random from the interval [-0.05, 0.05]. 
The training process lasted for 20k steps at a constant learning rate of 0.1, 
where stochastic gradient descent was employed with 5% out of the 60k
training examples. 
