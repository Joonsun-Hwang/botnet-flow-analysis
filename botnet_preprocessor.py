#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:16:09 2018

@author: junseon
"""

import os
from datetime import datetime, timedelta
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from itertools import combinations

# Set your working directory
MY_WORKING_DIRECTORY = os.getcwd()

# Set datetime string format
DATETIME_FORMAT = '%Y/%m/%d %H:%M:%S.%f'

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=self.test_size, 
                                 random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

class Botnet_Processor:
    
    def __init__(self, data):
        # Read csv file using loader
        self.raw_data = data
        self.preprocessed_data = pd.DataFrame()
        self.X_train = np.empty((0,0))
        self.X_test = np.empty((0,0))
        self.y_train = np.empty((0,0))
        self.y_test = np.empty((0,0))
    
    def get_head(self, n=5):
        return self.raw_data.head(n)
    
    @staticmethod
    def split_data_by_class(data):
        botnet = data[data['Label'].str.contains('Botnet')]
        background = data[data['Label'].str.contains('Background')]
        normal = data[data['Label'].str.contains('Normal')]
        
        dist = {'Background': background.shape[0], 'Botnet': botnet.shape[0], 'Normal': normal.shape[0]}
        
        return dist
    
    def get_distribution_columns(self):
        distribution_list = []
        for name, col in self.raw_data.iteritems():
            distribution_list.append(dict(col.value_counts()))
        return distribution_list
    
    def __treat_data__(self):
        # Convert the dataframe into ndarray
        ctu_nd = self.raw_data.values
        
        # Init dummy variable for pre-processing
        end_times = []
        
        src_well_known_ports = [] # < 1024
        src_registered_ports = [] # < 49152
        src_private_ports = [] # < 65536
        
        dst_well_known_ports = []
        dst_registered_ports = []
        dst_private_ports = []
        
        binary_label = []
        
        for a_row in ctu_nd:
            # Convert str into datetime
            a_row[0] = datetime.strptime(a_row[0], DATETIME_FORMAT)
            a_row[1] = "{:.6f}".format(a_row[1])
            a_row[1] = timedelta(seconds=int(a_row[1].split('.')[0]), microseconds=int(a_row[1].split('.')[1]))
            end_times.append(a_row[0] + a_row[1])
            a_row[1] = float(str(a_row[1].seconds) + '.' + str(a_row[1].microseconds))
            
            # Trim the direction column
            a_row[5] = a_row[5].strip()
            
            # Assign binary label
            if 'From-Botnet' in a_row[14]:
                if 'CC' in a_row[14] or 'DNS' in a_row[14]:
                    binary_label.append(0)
                else:
                    binary_label.append(1)
            else:
                binary_label.append(0)
            
            # If port number is hex number, convert into decimal number
            try:
                if any(c.isalpha() for c in a_row[4]):
                    a_row[4] = int(a_row[4], 16)
                else:
                    a_row[4] = int(a_row[4])
            except TypeError:
                if math.isnan(a_row[4]):
                    pass
                else:
                    print('An error occurred when changing to a decimal number.')
            
            try:
                if any(c.isalpha() for c in a_row[7]):
                    a_row[7] = int(a_row[7], 16)
                else:
                    a_row[7] = int(a_row[7])
            except TypeError:
                if math.isnan(a_row[7]):
                    pass
                else:
                    print('An error occurred when changing to a decimal number.')
            
            # Check source port number
            if a_row[4] < 1024:
                src_well_known_ports.append(1)
                src_registered_ports.append(0)
                src_private_ports.append(0)
            elif a_row[4] < 49152:
                src_well_known_ports.append(0)
                src_registered_ports.append(1)
                src_private_ports.append(0)
            elif a_row[4] < 65536:
                src_well_known_ports.append(0)
                src_registered_ports.append(0)
                src_private_ports.append(1)
            else:
                src_well_known_ports.append(0)
                src_registered_ports.append(0)
                src_private_ports.append(0)
                
            # Check destination port number
            if a_row[7] < 1024:
                dst_well_known_ports.append(1)
                dst_registered_ports.append(0)
                dst_private_ports.append(0)
            elif a_row[7] < 49152:
                dst_well_known_ports.append(0)
                dst_registered_ports.append(1)
                dst_private_ports.append(0)
            elif a_row[7] < 65536:
                dst_well_known_ports.append(0)
                dst_registered_ports.append(0)
                dst_private_ports.append(1)
            else:
                dst_well_known_ports.append(0)
                dst_registered_ports.append(0)
                dst_private_ports.append(1)
                
        # Concat all features
        treated_ctu_nd = np.c_[ctu_nd[:, 0:2], 
                               end_times,
                               ctu_nd[:, 2:5],
                               src_well_known_ports, 
                               src_registered_ports, 
                               src_private_ports, 
                               ctu_nd[:, 5:8],
                               dst_well_known_ports,
                               dst_registered_ports,
                               dst_private_ports,
                               ctu_nd[:, 8:15],
                               binary_label]
        colnames = ['StartTime', 'Dur', 'EndTime', 'Proto', 'SrcAddr', 'Sport',
           'SrcWellKnownPort', 'SrcRegisteredPort', 'SrcPrivatePort', 'Dir',
           'DstAddr', 'Dport', 'DstWellKnownPort', 'DstRegisteredPort',
           'DstPrivatePort', 'State', 'sTos', 'dTos', 'TotPkts', 'TotBytes',
           'SrcBytes', 'Label', 'BinaryLabel']
        
        treated_ctu = pd.DataFrame(treated_ctu_nd, columns=colnames)
        
        return treated_ctu
    
    def preprocess(self, test_size=0.3):
        now = datetime.now()
        
        print('Currently, data is being preprocessed. It may take some time.')
        
        treated_ctu = self.__treat_data__()
        treated_ctu_nd = treated_ctu.values
        
        # Apply One-hot Encoding to categorical variables
        one_hot_encoded_proto = pd.get_dummies(treated_ctu['Proto'])
        one_hot_encoded_proto_nd = one_hot_encoded_proto.values
        one_hot_encoded_dir = pd.get_dummies(treated_ctu['Dir'])
        one_hot_encoded_dir_nd = one_hot_encoded_dir.values
        one_hot_encoded_state = pd.get_dummies(treated_ctu['State'])
        one_hot_encoded_state_nd = one_hot_encoded_state.values
        
        # Create data for machine learning or deep learning
        # No sTos & dTos Columns, because they have NaN values
        ctu_for_ml_nd = np.c_[treated_ctu_nd[:, 1], 
                              one_hot_encoded_proto_nd,
                              treated_ctu_nd[:, 6:9], 
                              one_hot_encoded_dir_nd,
                              treated_ctu_nd[:, 12:15], 
                              one_hot_encoded_state_nd, 
                              treated_ctu_nd[:, 18:21], 
                              treated_ctu_nd[:, 22]]
        colnames = [treated_ctu.columns[1]] + list(one_hot_encoded_proto.columns) + list(treated_ctu.columns[6:9]) + list(one_hot_encoded_dir.columns) + list(treated_ctu.columns[12:15]) + list(one_hot_encoded_state.columns) + list(treated_ctu.columns[18:21]) + [treated_ctu.columns[22]]
        ctu_for_ml = pd.DataFrame(ctu_for_ml_nd, columns=colnames)
        self.preprocessed_data = ctu_for_ml
        
        # Split the data into train & test data
        X, y = ctu_for_ml_nd[:, :ctu_for_ml_nd.shape[1]-1], ctu_for_ml_nd[:, ctu_for_ml_nd.shape[1]-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')
                
        # Init nomalization method
        mms = MinMaxScaler()
        
        X_train = mms.fit_transform(X_train)
        X_test = mms.fit_transform(X_test)
        
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to preprocess.-----')
        
        # Check the columns that have NaNs
        print('Please, wait a minute for checking columns.')
        checksum = 0
        for name, col in ctu_for_ml.iteritems():
            if col.isnull().values.any():
                print(name, 'column has NaN values')
            else:
                checksum += 1
        if checksum - ctu_for_ml.shape[1] == 0:
            print("There aren't any columns have NaN values")
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
            
        return X_train, X_test, y_train, y_test

    def logistic_regression(self, X_train, y_train, X_test, y_test, penalty='l2', C=1):
        
        now = datetime.now()
        # Init LogisticRegression
        lr = LogisticRegression(penalty=penalty, C=C)
        
        # Learning
        print('Starting the learning of Logistic Regression')
        lr.fit(X_train, y_train)
        print('Training accuracy:', lr.score(X_train, y_train))
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to learn for Logistic Regression.-----')
        
        # Test
        print('Test accuracy:', lr.score(X_test, y_test))
        
        # Predict
        y_pred = lr.predict(X_test)
        self.__draw_confusion_matrix__(y_test, y_pred)
        
        return lr, y_pred
    
    def random_forest(self, X_train, y_train, X_test, y_test, n_estimators=10000, calculate_impoartances=False):
        
        now = datetime.now()
        
        # Init RandomRorest
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1)
        
        # Learning
        print('Starting the learning of Random Forest')
        forest.fit(X_train, y_train)
        print('Training accuracy:', forest.score(X_train, y_train))
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to learn for Random Forest.-----')
        
        # Test
        print('Test accuracy:', forest.score(X_test, y_test))
        
        # Predict
        y_pred = forest.predict(X_test)
        self.__draw_confusion_matrix__(y_test, y_pred)
        
        if calculate_impoartances:
            # Calculating the importance of each feature
            print("Starting the calculation of features' importance")
            colnames = self.preprocessed_data.columns[:self.preprocessed_data.shape[1]-1]
            importances = forest.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for f in range(self.X_train.shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30, colnames[indices[f]], importances[indices[f]]))
        
            return forest, y_pred, importances
        else:
            return forest, y_pred
        
    def support_vector_machine(self, X_train, y_train, X_test, y_test, C=1, gamma=0.1):
        now = datetime.now()
        
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=0)
        
        # Learning
        print('Starting the learning of SVM')
        svm.fit(X_train, y_train)
        print('Training accuracy:', svm.score(X_train, y_train))
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to learn for SVM.-----')
        
        # Test
        print('Test accuracy:', svm.score(X_test, y_test))
        
        # Predict
        y_pred = svm.predict(X_test)
        self.__draw_confucion_matrix__(y_test, y_pred)
        
        return svm, y_pred

    def __draw_confusion_matrix__(self, y_test, y_pred):
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()

    def feature_select(self, method, X_train, y_train):
        print('Starting the feature selection')
        
        now = datetime.now()
        
        sbs = SBS(method, k_features=1)
        sbs.fit(X_train, y_train)
        
        k_feat = [len(k) for k in sbs.subsets_]
        plt.plot(k_feat, sbs.scores_, marker='o')
        plt.ylim([0.5, 1.1])
        plt.ylabel('Accuracy')
        plt.xlabel('Number of featrues')
        plt.grid()
        plt.show()
        
        selected_features = list(sbs.subsets_[list(sbs.scores_).index(max(list(sbs.scores_)))])
        
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to select the features-----')
        
        return selected_features
        
    def feature_extract_pca(self, pca, X_train, X_test):
        print('Starting the feature extraction using pca')
        now = datetime.now()
        
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to the extracting the features-----')
        
        return X_train_pca, X_test_pca


if __name__ == "__main__":
    from botnet_data_loader import Botnet_Data_Loader as loader

    data_a = loader().botnet_data(sample_size=800000)
    data_d = loader().botnet_data(sample_size=800000, class_rate=0.5)
    
    # Greed search for n_components of PCA
    test_acc_list = []
    components = range(50, 250, 10)
    penalty = 'l1'
    C = 100
    
    botnet_processor = Botnet_Processor(data = data_d)
    print(botnet_processor.get_head(10))
    
    X_train, X_test, y_train, y_test = botnet_processor.preprocess()
    
    X_val = X_train[400000:, :]
    X_train = X_train[:400000, :]
    y_val = y_train[400000:]
    y_train = y_train[:400000]
        
    for component in components:
        print("*****************************")
        print('component:', component)
        # lr, y_pred = botnet_processor.logistic_regression(X_train, y_train, X_test, y_test, penalty, C)
                
        pca = PCA(n_components=component)
        X_train_pca, X_test_pca = botnet_processor.feature_extract_pca(pca, X_train, X_test)
        lr, y_pred = botnet_processor.logistic_regression(X_train_pca, y_train, X_test_pca, y_test, penalty, C)
        
        print("*****************************")
    
    # Test Logistic Regression
    penalty = 'l1'
    C_list = [1, 10, 100, 200]
    for idx, data in enumerate([data_a, data_d]):
        botnet_processor = Botnet_Processor(data = data)
        
        X_train, X_test, y_train, y_test = botnet_processor.preprocess()
        
        X_val = X_train[400000:, :]
        X_train = X_train[:400000, :]
        y_val = y_train[400000:]
        y_train = y_train[:400000]
        
        for C in C_list:
            print("*****************************")
            print("Data: ", idx)
            print("C: ", C)
    
            lr, y_pred, accuracy = botnet_processor.logistic_regression(X_train, y_train, X_test, y_test, penalty, C)
            
            pca = PCA(n_components=110)
            X_train_pca, X_test_pca = botnet_processor.feature_extract_pca(pca, X_train, X_test)
            lr, y_pred = botnet_processor.logistic_regression(X_train_pca, y_train, X_test_pca, y_test, penalty, C)
            
            print("*****************************")
            
    # Test SVM
    C_list = [1, 10, 100]
    gamma_list = [0.1, 1, 10]
    for idx, data in enumerate([data_a, data_d]):
        botnet_processor = Botnet_Processor(data = data)
        
        X_train, X_test, y_train, y_test = botnet_processor.preprocess()
        
        X_val = X_train[400000:, :]
        X_train = X_train[:400000, :]
        y_val = y_train[400000:]
        y_train = y_train[:400000]
        
        for C in C_list:
            for gamma in gamma_list:
                print("*****************************")
                print("Data: ", idx)
                print("C: ", C)
                print("gamma: ", gamma)
        
                lr, y_pred = botnet_processor.support_vector_machine(X_train, y_train, X_test, y_test, C, gamma)
                
                pca = PCA(n_components=110)
                X_train_pca, X_test_pca = botnet_processor.feature_extract_pca(pca, X_train, X_test)
                lr, y_pred = botnet_processor.support_vector_machine(X_train_pca, y_train, X_test_pca, y_test, C, gamma)
                
                print("*****************************")
                
    # Test Random Forest
    n_list = [10000]
    for idx, data in enumerate([data_a, data_d]):
        botnet_processor = Botnet_Processor(data = data)
        
        X_train, X_test, y_train, y_test = botnet_processor.preprocess()
        
        X_val = X_train[400000:, :]
        X_train = X_train[:400000, :]
        y_val = y_train[400000:]
        y_train = y_train[:400000]
        
        for n in n_list:
            print("*****************************")
            print("Data: ", idx)
            print("n: ", n)
    
            if idx == 1:
                lr, y_pred = botnet_processor.random_forest(X_train, y_train, X_test, y_test, n)
        
            pca = PCA(n_components=110)
            X_train_pca, X_test_pca = botnet_processor.feature_extract_pca(pca, X_train, X_test)
            lr, y_pred = botnet_processor.random_forest(X_train_pca, y_train, X_test_pca, y_test, n)
                
            print("*****************************")
    
    # Export data visualization & arena simulation
    data = loader().botnet_data(scenarios=[5, 7, 11])
    botnet_processor = Botnet_Processor(data = data)
    print(botnet_processor.get_head(10))
    treated_data = botnet_processor.__treat_data__()
    
    arena_data = treated_data[treated_data['BinaryLabel']==1]
    arena_data.to_csv('/Users/junseon/Desktop/arena.csv')
    