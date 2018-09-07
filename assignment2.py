# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:13:56 2017

@author: Aneeshaa S Chowdhry
"""
#Import pandas and numerical python libraries
import pandas as pd
#import numpy as np

#Import warnings library and suppress warnings displayed by python
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x:'%f'%x)
#Read data file
gapdata=pd.read_csv('gapminder.csv',low_memory=False)

#print number of rows and columns in file
print("Number of countries:", len(gapdata))

print("Number of columns:", len(gapdata.columns))
print(' ')
print(' ')

#check how many countries have missing values and hence cannot be considered for data driven decision making
print('Variable               No. of missing values')
print(gapdata.isnull().sum())
print(' ')
print(' ')

######################################################################################
#categorise alcohol consumption rates 
gapdata['alcconsumptionrange']=pd.qcut(gapdata.alcconsumption,4,labels=["very low","low","average","high"])

#generate frequency table on secondary variable
print('Displaying frequency distribution of Alcohol consumption range of countries')
print(gapdata['alcconsumptionrange'].value_counts(dropna=False))
print(' ')
print(' ')


#####################################################################################3
#create subset from main dataset by dropping all rows with null/missing values
sub=gapdata.dropna()

#categorize countries based on life expectancy by custom defining category boundaries
sub['lifeexpectancyrange']=pd.cut(sub.lifeexpectancy,[49,65,80,100])

#cross check if assignment is correct
#print(pd.crosstab(sub['lifeexpectancyrange'],sub['lifeexpectancy']))


#print(sub['lifeexpectancyrange'])
print('Displaying frequency distribution of life expectancy range of countries')
print(sub['lifeexpectancyrange'].value_counts())
print(' ')
print(' ')

#####################################################################################3
#categorize countries based on life expectancy by custom defining category boundaries
gapdata['co2emissionsrange']=pd.qcut(gapdata.co2emissions,4,labels=["very low","low","average","high"])


print('Displaying frequency distribution of co2 emissions range of countries')
print(gapdata['co2emissionsrange'].value_counts(dropna=False))
print(' ')
print(' ')

##################################################################################

#creating a secondary variable: total electricy consumption perperson in his/her lifetime
#assume a person uses same amount of electricity every year for the rest of his life
#total electricity a person consumes in his life in Kwh is lifeexpectancy*relectricperperson
sub['totaleconsumed']=sub['lifeexpectancy']*sub['relectricperperson']

#sort dataset in descending order to display country with persons consuming most electricity
sub = sub.sort_values(['totaleconsumed'], ascending=[False])

#display top 5 countries with persons consuming most electricity
print(sub[['country','totaleconsumed']].head(5))
print(' ')
print(' ')
