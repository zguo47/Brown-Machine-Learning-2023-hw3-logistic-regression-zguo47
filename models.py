#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

   Brown CS142, Spring 2020
'''
import random
import numpy as np


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # An extra row added for the bias
        self.alpha = 0.03  # DO NOT TUNE THIS PARAMETER
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        converge = False
        num_epochs = 0
        m, n = X.shape
        while converge != True:
            last_batch_loss = self.loss(X, Y)
            num_epochs += 1
            np.random.shuffle(X)
            for i in range((m*n)//self.batch_size-1):
                X_batch = X[i*self.batch_size: (i+1)*self.batch_size]
                Y_batch = Y[i*self.batch_size: (i+1)*self.batch_size]
                d_loss = np.zeros(self.weights.shape)
                for x, y in zip(X_batch, Y_batch):
                    for j in range(self.n_classes-1):
                        if y == j:
                            d_loss[j] += (softmax(self.weights * x)[j]-1)*x
                        else:
                            d_loss[j] += (softmax(self.weights * x)[j])*x
                self.weights -= (self.alpha*d_loss)/len(X)
            if abs(self.loss(X, Y) - last_batch_loss) < self.conv_threshold:
                break
        return num_epochs

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        loss = 0
        for x, y in zip(X, Y):
            for j in range(self.n_classes-1):
                if y == j:
                    loss += np.log(1/(1+np.exp(-x))[j])
        return -loss/self.n_classes

    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        labels = np.argmax(softmax(np.dot(self.weights, np.transpose(X))),axis=0)
        return labels

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        ypred = self.predict(X)
        return np.sum(ypred == Y)/len(Y)
