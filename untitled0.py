# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:17:31 2018
@author: 212585611
"""

import os
os.chdir('C:\\Users\\212458792\\Desktop')

import pandas as pd
import numpy as np
from datetime import date
from collections import Counter
from numpy import loadtxt
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x:'%f'%x)
 
spark_future_pax_data = pd.read_csv('input_for_future_script.csv', parse_dates = ['spark_pax_flight_future_final.flight_boarding_time'])
spark_future_pax_data.columns = [w.replace('spark_pax_flight_future_final.','') for w in spark_future_pax_data.columns]


spark_future_pax_data.to_csv('IP_Future_data.csv',index=False)
 