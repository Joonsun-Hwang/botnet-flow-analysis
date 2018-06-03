#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:50:06 2018

@author: junseon
"""

import os
import glob
import pandas as pd
from datetime import datetime

MY_WORKING_DIRECTORY = os.getcwd()

class Botnet_Data_Loader:
    
    def botnet_data(self, sample_size=None, class_rate=None):
        now = datetime.now()
        print('Currently, data is being read. It may take some time.')
        all_csv_files = glob.glob(os.path.join(MY_WORKING_DIRECTORY, 'CTU-13-Dataset/*.csv'))
        df = pd.concat((pd.read_csv(f) for f in all_csv_files))
        
        if class_rate:
            if class_rate < 0.3:
                print('You should enter a number greater than 0.3 in the class_rate parameter.')
                return
            elif class_rate > 1.0:
                print('You should enter a number smaller than 1.0 in the class_rate parameter.')
                return
            else:
                df = self.__sample_rate_conversion__(df, class_rate)
            
        if sample_size:
            if sample_size < 10000:
                print('You should enter a number greater than 10000 in the sample_size parameter.')
                return
            elif sample_size > df.shape[0]:
                print('You should enter a number smaller than', df.shape[0], 'in the sample_size parameter.')
                return
            else:
                df = self.__random_sampling__(df, sample_size)
        
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to load the data sets.-----')
        
        return df
    
    def __random_sampling__(self, df, sample_size):
        return df.sample(n=sample_size, replace=False)
        
    def __sample_rate_conversion__(self, df, class_rate):
        botnet = df[df['Label'].str.contains('Botnet')]
        not_botnet = df[-df['Label'].str.contains('Botnet')]
        
        not_botnet_sample_size = int((1-class_rate) * botnet.shape[0] / class_rate)
        not_botnet = self.__random_sampling__(not_botnet, not_botnet_sample_size)
        
        return pd.concat([botnet, not_botnet], ignore_index=True)
    