# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:18:45 2017

@author: 212458792
"""
import pandas as pd
import numpy as np
#to use ordinary least squares package
import statsmodels.api as sm
import statsmodels.formula.api as smf
#to remove warning messages
import warnings
warnings.filterwarnings("ignore")
#to plot data
import seaborn as sb
import matplotlib.pyplot as pt


#read data from csv file
gapdata=pd.read_csv('gapminder.csv')

# Convert variables to numeric
gapdata['polityscore']=gapdata['polityscore'].convert_objects(convert_numeric=True)
gapdata['urbanrate']=gapdata['urbanrate'].convert_objects(convert_numeric=True)
gapdata['lifeexpectancy']=gapdata['lifeexpectancy'].convert_objects(convert_numeric=True)


# Recoding Categorical variable
# All countries with polityscore<0 (autocratic) coded as 0 
# All democratic countries with polityscore>0 coded as 1\
gapdata['polityscorerange']=np.where((gapdata['polityscore']<0),0,1)
print('Frequency table for the recoded variable: polityscorerange')
print(gapdata['polityscorerange'].value_counts())
print('  ')
print('  ')

print('Fig1 - Linear regression model for polityscorerange vs life expectancy')
mod1=smf.ols('lifeexpectancy~polityscorerange',data=gapdata).fit()
print(mod1.summary())
print('  ')
print('  ')


print('Fig2 - Linear regression model for urbanrate vs life expectancy')
mod1=smf.ols('lifeexpectancy~urbanrate',data=gapdata).fit()
print(mod1.summary())

print('  ')
print('  ')
print('Describing urbanrate')
print(gapdata['urbanrate'].describe())
#sb.pairplot(gapdata, x_vars=['urbanrate','incomeperperson','employrate','polityscore'], y_vars='lifeexpectancy', size=4, aspect=0.7, kind='reg')
