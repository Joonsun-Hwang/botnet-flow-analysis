#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:51:11 2018

@author: junseon
"""

# TODO: ANN, DNN

import functools

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt

from botnet_data_loader import Botnet_Data_Loader as loader
from botnet_preprocessor import Botnet_Processor as processor

class Neural_Network:

    def __init__(self, input_dim, output_dim=1, epochs=1000, batch_size=128, hidden_layer=4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer = hidden_layer-1
        self.first_hidden_node = 258
        
        self.model = Sequential()
        
    def build_model(self):
        self.model.add(Dense(int(self.first_hidden_node), activation='relu', input_dim=self.input_dim))
        self.model.add(Dropout(0.5))
        
        for i in range(self.hidden_layer):
            self.model.add(Dense(int(self.first_hidden_node/(i+2)), activation='relu'))
            self.model.add(Dropout(0.5))
        
        self.model.add(Dense(self.output_dim, activation='sigmoid'))
        
        precision = self.__as_keras_metric__(tf.metrics.precision)
        recall = self.__as_keras_metric__(tf.metrics.recall)
        self.model.compile(optimizer='sgd',
                           loss='binary_crossentropy',
                           metrics=['accuracy', precision, recall])
        
    def fit(self, X_train, y_train, X_val, y_val):
        hist = self.model.fit(X_train, y_train, 
                              epochs=self.epochs, batch_size=self.batch_size, 
                              validation_data=(X_val, y_val))
        
        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()
        
        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        
        loss_ax.plot(hist.history['acc'], 'b', label='train acc')
        loss_ax.plot(hist.history['val_acc'], 'g', label='val acc')
        
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')
        
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        
        plt.show()
        
        return hist
    
    @staticmethod
    def __as_keras_metric__(method):
        @functools.wraps(method)
        def wrapper(self, args, **kwargs):
            """ Wrapper for turning tensorflow metrics into keras metrics """
            value, update_op = method(self, args, **kwargs)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
            return value
        return wrapper


if __name__ == "__main__":
    data = loader.botnet_data(sample_size=100000)
    
    botnet_processor = processor(data = data)
    botnet_processor.print_head(10)
    
    X_train, X_test, y_train, y_test = botnet_processor.preprocess()
    
    X_val = X_train[50000:, :]
    X_train = X_train[:50000, :]
    y_val = y_train[50000:]
    y_train = y_train[:50000]
    
    nn = Neural_Network(input_dim=X_train.shape[1])
    nn.build_model()
    nn.fit(X_train, y_train, X_val, y_val)
    
    lr, y_pred = botnet_processor.logistic_regression()

