# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:14:19 2017

@author: 212458792
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

#read data from csv file
student=pd.read_csv('student-mat.csv',low_memory=False)
#convert column names to upper case
student.columns=map(str.upper,student.columns)


#data management
recode1={'yes':1,'no':0}
student['ROMANTIC']=student['ROMANTIC'].map(recode1)
student['INTERNET']=student['INTERNET'].map(recode1)
student['HIGHER']=student['HIGHER'].map(recode1)
student['NURSERY']=student['NURSERY'].map(recode1)
student['ACTIVITIES']=student['ACTIVITIES'].map(recode1)
student['PAID']=student['PAID'].map(recode1)
student['FAMSUP']=student['FAMSUP'].map(recode1)
student['SCHOOLSUP']=student['SCHOOLSUP'].map(recode1)

recode2={'mother':1,'father':0,'other':0}
student['GUARDIAN_MOTH']=student['GUARDIAN'].map(recode2)

recode21={'mother':0,'father':1,'other':0}
student['GUARDIAN_FATH']=student['GUARDIAN'].map(recode21)

recode22={'mother':0,'father':0,'other':1}
student['GUARDIAN_OTHR']=student['GUARDIAN'].map(recode22)

recode3={'U':1,'R':0}
student['ADDRESS']=student['ADDRESS'].map(recode3)

recode4={'GP':1,'MS':0}
student['SCHOOL']=student['SCHOOL'].map(recode4)

recode5={'F':1,'M':0}
student['SEX']=student['SEX'].map(recode5)

recode6={'GT3':1,'LE3':0}
student['FAMSIZE']=student['FAMSIZE'].map(recode6)

recode7={'T':1,'A':0}
student['PSTATUS']=student['PSTATUS'].map(recode7)

recode8={'at_home':1,'health':0,'other':0,'services':0,'teacher':0}
student['MJOB_ATHOME']=student['MJOB'].map(recode8)
student['FJOB_ATHOME']=student['FJOB'].map(recode8)

recode81={'at_home':0,'health':1,'other':0,'services':0,'teacher':0}
student['MJOB_HEALTH']=student['MJOB'].map(recode81)
student['FJOB_HEALTH']=student['FJOB'].map(recode81)

recode82={'at_home':0,'health':0,'other':1,'services':0,'teacher':0}
student['MJOB_OTHR']=student['MJOB'].map(recode82)
student['FJOB_OTHR']=student['FJOB'].map(recode82)

recode83={'at_home':0,'health':0,'other':0,'services':1,'teacher':0}
student['MJOB_SERV']=student['MJOB'].map(recode83)
student['FJOB_SERV']=student['FJOB'].map(recode83)

recode84={'at_home':0,'health':0,'other':0,'services':0,'teacher':1}
student['MJOB_TEACH']=student['MJOB'].map(recode84)
student['FJOB_TEACH']=student['FJOB'].map(recode84)

recode9={'course':1,'home':0,'other':0,'reputation':0}
student['REASON_COURSE']=student['REASON'].map(recode9)

recode91={'course':0,'home':1,'other':0,'reputation':0}
student['REASON_HOME']=student['REASON'].map(recode91)

recode92={'course':0,'home':0,'other':1,'reputation':0}
student['REASON_OTHR']=student['REASON'].map(recode92)

recode93={'course':0,'home':0,'other':0,'reputation':1}
student['REASON_REP']=student['REASON'].map(recode93)


predictors=student.copy()

predictors.drop(['MJOB','FJOB','REASON','GUARDIAN','G3'], axis=1, inplace=True)

# variables standardization
from sklearn import preprocessing
for col in predictors.columns:
    predictors[col] = preprocessing.scale(predictors[col].astype('float64'))

target=student.G3

#data set split
pred_train,pred_test,tar_train,tar_test=train_test_split(predictors,target,test_size=.3,random_state=123)


model=LassoLarsCV(cv=10,precompute=False).fit(pred_train,tar_train)

#plot coeffs
c=dict(zip(predictors.columns, model.coef_))

plt.bar(range(len(c)), c.values(), align='center',color='b')
plt.axhline(y=0, color='k', linestyle='-',linewidth=1)
plt.ylabel('Regression Coefficients')
plt.xlabel('Variables')
plt.xticks(range(len(c)), list(c.keys()))
plt.xticks(rotation=90,fontsize=7)
plt.show()


# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('Training Data MSE')
print('~~~~~~~~~~~~~~~~~~~~')
print(train_error)
print(' ')
print(' ')

print ('Test Data MSE')
print('~~~~~~~~~~~~~~~~')
print(test_error)

print(' ')
print(' ')
# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print('Training Data R-square')
print('~~~~~~~~~~~~~~~~~~~~~~~')
print(rsquared_train)
print(' ')
print(' ')

print ('Test Data R-square')
print('~~~~~~~~~~~~~~~~~~~~')
print(rsquared_test)
print(' ')
print(' ')


""" 
for col in student.columns:
    print(student[col].value_counts(dropna=False).reset_index())
"""