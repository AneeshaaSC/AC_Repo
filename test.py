# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:41:35 2018

@author: 212458792
"""
import pandas as pd
import numpy as np


df = pd.DataFrame({'id' : [1,1,1,2,2,3,3,3,3,4,4,5,6,6,6,7,7],
                   'value'  : [np.nan,"second","second","first",
                               "second","first","third","fourth",
                               "fifth","second","fifth","first",
                               "first","second","third","fourth","fifth"]})
#print(df)
l=df.groupby('id').first()

 

print(l)

grouped_df = df.sort_values('value').groupby('id')

for key, item in grouped_df:
    print(grouped_df.get_group(key)), "\n\n"