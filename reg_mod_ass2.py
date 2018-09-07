# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:21:14 2017

@author: 212458792
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sb


# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

# read data file
gapdata = pd.read_csv('gapminder.csv')

# get rows with valid values only
sub1 = gapdata[['hivrate','incomeperperson', 'lifeexpectancy','internetuserate','alcconsumption']].dropna()

# convert to numeric format
sub1['hivrate'] = pd.to_numeric(sub1['hivrate'], errors='coerce')
sub1['internetuserate'] = pd.to_numeric(sub1['internetuserate'], errors='coerce')
sub1['alcconsumption'] = pd.to_numeric(sub1['alcconsumption'], errors='coerce')
sub1['lifeexpectancy'] = pd.to_numeric(sub1['lifeexpectancy'], errors='coerce')
sub1['incomeperperson'] = pd.to_numeric(sub1['incomeperperson'], errors='coerce')
#sub1['urbanrate'] = pd.to_numeric(sub1['urbanrate'], errors='coerce')
#sub1['employrate'] = pd.to_numeric(sub1['employrate'], errors='coerce')


# centering variables
sub1['alcconsumption_c'] = (sub1['alcconsumption'] - sub1['alcconsumption'].mean())
sub1['hivrate_c'] = (sub1['hivrate'] - sub1['hivrate'].mean())#3
sub1['internetuserate_c'] = (sub1['internetuserate'] - sub1['internetuserate'].mean())
sub1['incomeperperson_c'] = (sub1['incomeperperson'] - sub1['incomeperperson'].mean())
#sub1['urbanrate_c'] = (sub1['urbanrate'] - sub1['urbanrate'].mean())
#sub1['employrate_c'] = (sub1['employrate'] - sub1['employrate'].mean())

#####################################################################################
#   TEST FOR CONFOUNDING
#####################################################################################

#employrate is confounded by urbanrate
#reg2 = smf.ols('lifeexpectancy ~ employrate_c', data=sub1).fit()
#print (reg2.summary())

#reg2 = smf.ols('lifeexpectancy ~ urbanrate_c+employrate_c', data=sub1).fit()
#print (reg2.summary())

####################################################################################

####################################################################################
# POLYNOMIAL REGRESSION
####################################################################################

#first order (linear) scatterplot
#scat1 = sb.regplot(x="hivrate_c", y="lifeexpectancy", scatter=True,data=sub1)
#plt.xlabel('alcohol consumption')
#plt.ylabel('life expectancy')

#scat1 = sb.regplot(x="hivrate_c", y="lifeexpectancy", scatter=True,order=2,data=sub1)
#plt.xlabel('alcohol consumption')
#plt.ylabel('life expectancy')
#reg1 = smf.ols('lifeexpectancy ~ employrate_c+internetuserate_c+hivrate_c+alcconsumption_c', data=sub1).fit()
#print (reg1.summary())

reg1 = smf.ols('lifeexpectancy ~ incomeperperson_c+I(incomeperperson_c**2)+internetuserate_c+I(internetuserate_c**2)+hivrate_c+I(hivrate_c**2)', data=sub1).fit()
print (reg1.summary())

#incomeperperson_c+I(incomeperperson_c**2)+internetuserate_c+I(internetuserate_c**2)+hivrate_c+I(hivrate_c**2)
#incomeperperson_c+urbanrate_c+hivrate_c+internetuserate_c--68% -- residuals less than 3
#employrate_c+internetuserate_c+hivrate_c+incomeperperson+I(alcconsumption_c**2)--74.7% -- has outlier
#employrate_c+internetuserate_c+hivrate_c+alcconsumption_c --75.8% -- has outliers
#employrate_c+internetuserate_c+hivrate_c+I(alcconsumption_c**2)--77.2%  -- has outliers

####################################################################################
# DIAGNOSTIC PLOTS
####################################################################################

#Q-Q plot for normality
fig1=sm.qqplot(reg1.resid, line='r')

# simple plot of residuals
#stdres=pd.DataFrame(reg1.resid_pearson)
#plt.plot(stdres, 'o', ls='None')
#l = plt.axhline(y=0, color='r')
#plt.ylabel('Standardized Residual')
#plt.xlabel('Observation Number')


# additional regression diagnostic plots

#fig1 = sm.graphics.plot_regress_exog(reg1,  "incomeperperson_c")
#fig2 = sm.graphics.plot_regress_exog(reg1,  "internetuserate_c")
#fig3 = sm.graphics.plot_regress_exog(reg1,  "hivrate_c")


# leverage plot
#fig3=sm.graphics.influence_plot(reg1, size=8)
#print(fig3)


#sb.pairplot(gapdata, x_vars=['urbanrate','incomeperperson','employrate','polityscore','hivrate','co2emissions','suicideper100th','breastcancerper100th','alcconsumption'], y_vars='lifeexpectancy', size=4, aspect=1, kind='reg')



 

