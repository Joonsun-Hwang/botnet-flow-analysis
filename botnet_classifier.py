#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:51:11 2018

@author: junseon
"""

import functools

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class Neural_Network:

    def __init__(self, input_dim, output_dim=1, learning_rate=0.00003, epochs=10000, batch_size=256, hidden_layer=10, first_hidden_node=258, batch_normalization=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer = hidden_layer-1
        self.first_hidden_node = first_hidden_node
        
        self.model = Sequential()
        
        self.batch_normalization = batch_normalization
        
    def build_model(self):
        if self.batch_normalization:
            # Input Layer
            self.model.add(Dense(int(self.first_hidden_node), input_dim=self.input_dim))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            
            # Hidden Layer
            for i in range(self.hidden_layer):
                self.model.add(Dense(int(self.first_hidden_node/(i+2))))
                self.model.add(BatchNormalization())
                self.model.add(Activation('relu'))
                self.model.add(Dropout(0.5))
            
            # Output Layer
            self.model.add(Dense(self.output_dim))
            self.model.add(BatchNormalization())
            self.model.add(Activation('sigmoid'))
        else:
            # Input Layer
            self.model.add(Dense(int(self.first_hidden_node), input_dim=self.input_dim))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            
            # Hidden Layer
            for i in range(self.hidden_layer):
                self.model.add(Dense(int(self.first_hidden_node/(i+2))))
                self.model.add(Activation('relu'))
                self.model.add(Dropout(0.5))
            
            # Output Layer
            self.model.add(Dense(self.output_dim))
            self.model.add(Activation('sigmoid'))
        
        adam = Adam(lr=self.learning_rate)
        
        self.model.compile(optimizer=adam,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
    def fit(self, X_train, y_train, X_val, y_val):
        early_stopping = EarlyStopping(patience=10)
        hist = self.model.fit(X_train, y_train, 
                              epochs=self.epochs, batch_size=self.batch_size, 
                              validation_data=(X_val, y_val),
                              callbacks=[early_stopping])
        
        fig, loss_ax = plt.subplots(figsize=(8,6))

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
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64)
    
    def predict(self, X_test, y_test):
        y_pred = self.model.predict_classes(X_test)
        
        self.__draw_confusion_matrix__(y_test, y_pred)
        
        return y_pred
    
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

    def __draw_confusion_matrix__(self, y_test, y_pred):
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()
    
    
if __name__ == "__main__":
    from botnet_data_loader import Botnet_Data_Loader as loader
    from botnet_preprocessor import Botnet_Processor as processor

    data = loader().botnet_data(sample_size=800000, class_rate=0.5)
    
    botnet_processor = processor(data=data)
    botnet_processor.get_head(10)
    
    X_train, X_test, y_train, y_test = botnet_processor.preprocess()
    
    X_val = X_train[400000:, :]
    X_train = X_train[:400000, :]
    y_val = y_train[400000:]
    y_train = y_train[:400000]
    
    nn = Neural_Network(input_dim=X_train.shape[1], hidden_layer=30, learning_rate=0.0003)
    nn.build_model()
    nn.fit(X_train, y_train, X_val, y_val)
    
    print(nn.evaluate(X_test, y_test))
    y_pred = nn.predict(X_test, y_test)
    
    # [loss, accuracy, precision, recall]
    
    # 20 hidden layers, lr = 0.00003
    # [0.16625044167637826, 0.9583875, 0.8767687260105649, 0.9962718793182865]

    # 20 hidden layers + batch normalization, lr = 0.00003
    # [0.6413604638814926, 0.6710791666666667, 0, 0]
    
    # 30 hidden layers, lr = 0.0003
    # [0.17705267111559708, 0.9587958333333333, 0.8781241686829404, 0.9961487612806805]
    
    # 35 hidden layers, lr = 0.0003, batch size = 256
    # []
    
    # 40 hidden layers, lr = 0.001, batch size = 256
    # [0.2374726113875707, 0.952475]
    
    # 50 hidden layers, lr = 0.0003
    # 
    
    # 50 hidden layers + batch normalization, lr = 0.00003
    # [0.6413604638814926, 0.6710791666666667, 0, 0]
