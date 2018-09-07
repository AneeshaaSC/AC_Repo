# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:13:56 2017

@author: Aneeshaa S Chowdhry
"""
#Import pandas and numerical python libraries
import pandas as pd
import numpy as np

#Import warnings library and suppress warnings displayed by python
import warnings
warnings.filterwarnings("ignore")

#Read data file
gapdata=pd.read_csv('gapminder.csv',low_memory=False)

#print number of rows and columns in file
print("Number of countries:", len(gapdata))

print("Number of columns:", len(gapdata.columns))
print(' ')
print(' ')

####################################################################################
#convert polityscore column datatype to numeric, its currently a string
gapdata['polityscore']=gapdata['polityscore'].convert_objects(convert_numeric=True)

#assign range to polityscore
gapdata['polityscorerange'] = np.where((gapdata['polityscore']<=-4)&(gapdata['polityscore']<=-10), 'low democracy',np.where((gapdata['polityscore']>=-4)&(gapdata['polityscore']<4), 'average democracy',np.where((gapdata['polityscore']>=4)&(gapdata['polityscore']<10), 'high democracy','NAN')))

#count number of countries in various polityscore ranges
a=gapdata["polityscorerange"].value_counts()
print('Number of countries in various Polity score or Democracy ranges')
print(a)
print(' ')
print(' ')
####################################################################################
#convert internetuserate column datatype to numeric, its currently a string
gapdata['internetuserate']=gapdata['internetuserate'].convert_objects(convert_numeric=True)

#assign range to internetuserate values
gapdata['internetuserange'] = np.where((gapdata['internetuserate']>=85), 'very high',np.where((gapdata['internetuserate']>=70)&(gapdata['internetuserate']<85), 'high',np.where((gapdata['internetuserate']>=55)&(gapdata['internetuserate']<70), 'average',np.where((gapdata['internetuserate']>=40)&(gapdata['internetuserate']<55), 'low','very low'))))

#count number of countries in various internet usage ranges
b=gapdata['internetuserange'].value_counts(dropna=False)
print('Number of countries in various Internet Usage Rate ranges')
print(b)
print(' ')
print(' ')

#########################################################################################
#convert employrate column datatype to numeric, its currently a string
gapdata['employrate']=gapdata['employrate'].convert_objects(convert_numeric=True)

#assign range to employrate values
gapdata['employrate'] = np.where(gapdata['employrate']>=85, 'very high',np.where((gapdata['employrate']>=70)&(gapdata['employrate']<85), 'high',np.where((gapdata['employrate']>=55)&(gapdata['employrate']<70), 'average',np.where((gapdata['employrate']>=40)&(gapdata['employrate']<55), 'low','very low'))))

#count number of countries in various employrate ranges
c=gapdata['employrate'].value_counts()
print('Number of countries in various Employment Rate ranges')
print(c)
print(' ')
print(' ')

###############################################################################################

#convert urbanrate column datatype to numeric, its currently a string
gapdata['urbanrate']=gapdata['urbanrate'].convert_objects(convert_numeric=True)

#assign range to urbanrate values
gapdata['urbanrate'] = np.where(gapdata['urbanrate']>=85, 'very high',np.where((gapdata['urbanrate']>=70)&(gapdata['urbanrate']<85), 'high',np.where((gapdata['urbanrate']>=55)&(gapdata['urbanrate']<70), 'average',np.where((gapdata['urbanrate']>=40)&(gapdata['urbanrate']<55), 'low','very low'))))

#count number of countries in various employrate ranges
d=gapdata['urbanrate'].value_counts()
print('Number of countries in various Urban Rate ranges')
print(d)
print(' ')
print(' ')

e=gapdata.groupby('internetuserange').size()*100/len(gapdata)
print(e)

##############################################################################################

selectdata=gapdata[(gapdata['country']=='India')]
print(selectdata)