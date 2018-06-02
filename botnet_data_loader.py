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

class botnet_data_loader:
    
    # TODO: If you need up-sampling or down-sampling, develop the parameter class_rate
    def botnet_data(sample_size, class_rate=0):
        now = datetime.now()
        print('Currently, data is being read. It may take some time.')
        MY_WORKING_DIRECTORY = os.getcwd()
        all_csv_files = glob.glob(os.path.join(MY_WORKING_DIRECTORY, 'CTU-13-Dataset/*.csv'))
        df = pd.concat((pd.read_csv(f) for f in all_csv_files))
        
        if sample_size:
            df = df.sample(n=sample_size, replace=False)
        
        duration = datetime.now() - now
        print('----- It took ' + str(duration.seconds) + '.' + str(duration.microseconds) + ' seconds to load the data sets.-----')
        
        return df
