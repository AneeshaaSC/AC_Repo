# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:06:52 2017

"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot  as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# read data file
nesarc=pd.read_csv('nesarc_pds.csv')

nesarc.rename(columns={'S4AQ4A16':'suicidal'},inplace=True)

nesarc['suicidal']=nesarc['suicidal'].replace(9, np.nan)
nesarc=nesarc.dropna()
 
def uglyyesno(p):
    if p==2:
        return 0
    else:
        return 1

nesarc['suicidal']=nesarc['suicidal'].apply(lambda p: uglyyesno(p))
   


print(nesarc['MAJORDEPLIFE'].value_counts(dropna=False))
print(nesarc['suicidal'].value_counts(dropna=False))


def correl (row):
   if row['suicidal']==1 and row['MAJORDEPLIFE'] == 1 :
      return 'suicidal due to dep'
   elif row['suicidal']==1 and row['MAJORDEPLIFE'] == 0 :
      return 'suicidal but not dep'
   elif row['suicidal']==0 and row['MAJORDEPLIFE'] == 1 :
      return 'other depression'
   else:
      return 'happy'
  
  
nesarc['hyptrue'] = nesarc.apply (lambda row: correl (row),axis=1)

print(nesarc['hyptrue'].value_counts(dropna=False))

#ax = sb.countplot(x="MAJORDEPLIFE", hue="unpretty", data=nesarc)
bx = sb.countplot(x="hyptrue", data=nesarc)

#S10Q1A33--fidelity
#S10Q1A54--attention seeking
#S11AQ1A4 -- runaway
"""
nesarc.boxplot(column="unpretty",        # Column to plot
                 by= "MAJORDEPLIFE",         # Column to split upon
                 figsize= (8,8))        # Figure size
"""
#S10Q1A54
#S11AQ1A17

#S4AQ4A1--maj dep
#S4AQ4A16