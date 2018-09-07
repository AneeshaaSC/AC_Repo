# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:09:04 2017

@author: 212458792
"""

#Import pandas and numerical python libraries
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pt
#import numpy as np

#Format the display a little bit
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x:'%f'%x)

#Import warnings library and suppress warnings displayed by python
import warnings
warnings.filterwarnings("ignore")

#Read data file
gapdata=pd.read_csv('gapminder.csv',low_memory=False)

#plot univariate graph of life expectance range 
gapdata['lifeexpectancyrange']=pd.cut(gapdata.lifeexpectancy,[0,49,65,80,100])
sb.countplot(x='lifeexpectancyrange',data=gapdata)
print(' ')
print(' ')

gapdata = gapdata.sort_values(['lifeexpectancyrange'], ascending=[False])

#display top 5 countries with persons consuming most electricity
print('Top 5 countries to move to if you want to live longer')
print(gapdata[['country','lifeexpectancyrange']].head(5))
print(' ')
print(' ')


#remove all rows with null data in any of the variables
sub=gapdata.dropna()


#find country with highest alcohol consumption rate
#g=sb.factorplot(x='country',y='alcconsumption',data=sub,kind='bar',ci=None,aspect=1.85)
#g.set_xticklabels(rotation=90)


# plot bivariate graph of urban rate vs life expectancy
#sb.regplot(x='urbanrate',y='lifeexpectancy',data=sub)
#pt.title('Relationship Between Urban Rate and Life Expectancy')

# plot bivariate graph of income per person vs life expectancy
sb.regplot(x='incomeperperson',y='lifeexpectancy',data=sub)
#pt.title('Relationship Between Income Per Person and Life Expectancy')






