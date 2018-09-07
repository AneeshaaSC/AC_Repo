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
gapdata=pd.read_csv('gapminder.csv')

# get observations with not null values in required attributes
sub=gapdata[['polityscore','femaleemployrate','alcconsumption','armedforcesrate']].dropna()

#function to recode polityscore variable 
def polityscore_recode(p):
    if p<0:
        return 'Autocracy'
    else:
        return 'Democracy'

#apply above function to polityscore column
sub['freedom']=sub['polityscore'].apply(lambda p: polityscore_recode(p))

#convert variables to numeric
sub['polityscore']=sub['polityscore'].convert_objects(convert_numeric=True)
sub['alcconsumption']=sub['alcconsumption'].convert_objects(convert_numeric=True)
sub['armedforcesrate']=sub['armedforcesrate'].convert_objects(convert_numeric=True)
#sub['hivrate']=sub['hivrate'].convert_objects(convert_numeric=True)

#no effect of internetuserate, co2emissions, hivrate,femaleemployrate,oilperperson,relectricperperson
# ,lifeexpectancy, suicideper100th, urbanrate on polityscore
#fig1=sb.regplot(x="polityscore",y="armedforcesrate",order=4,data=sub)
#fig2=sb.factorplot(x='freedom',y='alcconsumption',data=sub,kind="bar",ci=None)

recode1={'Autocracy': 0, 'Democracy': 1}
sub['new_polityscore']=sub['freedom'].map(recode1)
sub['new_polityscore']=sub['new_polityscore'].convert_objects(convert_numeric=True)


#print(sub['freedom'].value_counts())

sub.rename(columns={'new_polityscore':'freedomtodrink'},inplace=True)

#print(sub['freedomtodrink'].value_counts(dropna=False))
sub['alcconsumption_c']=sub['alcconsumption']-sub['alcconsumption'].mean()
sub['armedforcesrate_c']=sub['armedforcesrate']-sub['armedforcesrate'].mean()
#sub['hivrate_c']=sub['hivrate']-sub['hivrate'].mean()

#incomeperperson p value 0.068, employrate 0.181, urbanrate_c=0.362
#reg1=smf.logit(formula='freedomtodrink ~ alcconsumption_c',data=sub).fit()
#reg1=smf.logit(formula='freedomtodrink ~ hivrate_c',data=sub).fit()
reg1=smf.logit(formula='freedomtodrink ~ alcconsumption_c+armedforcesrate_c',data=sub).fit()
print(reg1.summary())
print(' ')
print(' ')
print('Odds Ratio')
print('-----------')
#print(np.exp(reg1.params)) 


params = reg1.params
conf = reg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))
#reg2=smf.logit(formula='freedomtodrink ~ alcconsumption_c+armedforcesrate_c',data=sub).fit()
#print(reg2.summary())

#print(' ')
#print(' ')
#print('Odds Ratio')
#print('-----------')
#print(np.exp(reg1.params)) 

"""
a=np.array(['foo', 'foo', 'foo', 'foo', 'ba r', 'bar',
       'bar', 'bar', 'foo', 'foo', 'foo'], dtype=object)

b=np.array(['one', 'one', 'one', 'two', 'one', 'one',
       'one', 'two', 'two', 'two', 'one'], dtype=object)

c=np.array(['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny',
       'shiny', 'dull', 'shiny', 'shiny', 'shiny'], dtype=object)
    
print(pd.crosstab(a, [b,c]))
"""

#plt.plot( sub['alcconsumption'], sub['freedomtodrink'] == 0, 'ro')
#plt.plot(sub['alcconsumption'],sub['freedomtodrink'] == 1,  'bx')

