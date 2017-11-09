# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:26:03 2017

@author: 212458792
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore")
 # Feature Importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import datasets
import matplotlib.pylab as plt 
import seaborn as sb

nesarc=pd.read_csv('nesarc_pds.csv',low_memory=False)

process_data = nesarc[['S2AQ7B','S1Q2310','S4AQ4A18','S4AQ4A17','S13Q5','S10Q1A14','S10Q1A15','OBCOMDX2','S10Q1A9','S4AQ4A16','GENAXDX12','S10Q1A5','SEX','MAJORDEPLIFE','S10Q1A12','S10Q1A21','S5Q6A9']]


#rename columns for convenience
process_data.rename(columns={'S4AQ4A16':'suicidal'},inplace=True)
process_data.rename(columns={'S10Q1A9':'dependence'},inplace=True)
process_data.rename(columns={'GENAXDX12':'anxiety'},inplace=True)
process_data.rename(columns={'S10Q1A5':'unpretty'},inplace=True)
process_data.rename(columns={'S10Q1A12':'peoplepleasing'},inplace=True)
process_data.rename(columns={'S10Q1A21':'hoarding'},inplace=True)
process_data.rename(columns={'OBCOMDX2':'ocd'},inplace=True)
process_data.rename(columns={'S10Q1A14':'needy'},inplace=True)
process_data.rename(columns={'S10Q1A15':'fearlonliness'},inplace=True)
process_data.rename(columns={'S13Q5':'crimevictim'},inplace=True)
process_data.rename(columns={'S5Q6A9':'settlingdown'},inplace=True)
process_data.rename(columns={'S4AQ4A18':'feellikedying'},inplace=True)
process_data.rename(columns={'S4AQ4A17':'wannasuicide'},inplace=True)
process_data.rename(columns={'S1Q2310':'debtridden'},inplace=True)
process_data.rename(columns={'S2AQ7B':'drinkingfreq'},inplace=True)

# Exclude unknowns from analysis- data management
process_data['needy']=process_data['needy'].replace(9, np.nan) 
process_data['fearlonliness']=process_data['fearlonliness'].replace(9, np.nan)
process_data['unpretty']=process_data['unpretty'].replace(9, np.nan)
process_data['debtridden']=process_data['debtridden'].replace(9, np.nan)
process_data['settlingdown']=process_data['settlingdown'].replace('BL', np.nan)
process_data['settlingdown'] =pd.to_numeric(process_data['settlingdown'], errors='coerce')
process_data['settlingdown']=process_data['settlingdown'].replace(9, np.nan)

process_data['feellikedying']=process_data['feellikedying'].replace('BL', np.nan)
process_data['feellikedying'] =pd.to_numeric(process_data['feellikedying'], errors='coerce')

process_data['feellikedying']=process_data['feellikedying'].replace(9, np.nan)
process_data['drinkingfreq'] =process_data['drinkingfreq'].replace('BL', np.nan)
process_data['drinkingfreq'] =pd.to_numeric(process_data['drinkingfreq'], errors='coerce')
process_data['drinkingfreq'] =process_data['drinkingfreq'].replace(99, np.nan)
process_data['wannasuicide']=process_data['wannasuicide'].replace('BL', np.nan)
process_data['wannasuicide'] =pd.to_numeric(process_data['wannasuicide'], errors='coerce')
process_data['wannasuicide']=process_data['wannasuicide'].replace(9, np.nan)
process_data['dependence']=process_data['dependence'].replace(9, np.nan)

process_data['suicidal']=process_data['suicidal'].replace('BL', np.nan)
process_data['suicidal'] =pd.to_numeric(process_data['suicidal'], errors='coerce')
process_data['suicidal']=process_data['suicidal'].replace(9, np.nan)
process_data['anxiety'] =process_data['anxiety'].replace(9, np.nan)
process_data['peoplepleasing'] =process_data['peoplepleasing'].replace(9, np.nan)
process_data['hoarding'] =process_data['hoarding'].replace(9, np.nan)
process_data['crimevictim'] =process_data['crimevictim'].replace(99, np.nan)


# convert variables to numeric
process_data['needy'] =pd.to_numeric(process_data['needy'], errors='coerce')
process_data['suicidal'] =pd.to_numeric(process_data['suicidal'], errors='coerce')
process_data['unpretty'] =pd.to_numeric(process_data['unpretty'], errors='coerce')
process_data['anxiety'] =pd.to_numeric(process_data['anxiety'], errors='coerce') 
process_data['SEX'] =pd.to_numeric(process_data['SEX'], errors='coerce') 
process_data['MAJORDEPLIFE'] =pd.to_numeric(process_data['MAJORDEPLIFE'], errors='coerce') 
process_data['peoplepleasing'] =pd.to_numeric(process_data['peoplepleasing'], errors='coerce') 
process_data['dependence'] =pd.to_numeric(process_data['dependence'], errors='coerce')
process_data['hoarding'] =pd.to_numeric(process_data['hoarding'], errors='coerce') 
process_data['crimevictim'] =pd.to_numeric(process_data['crimevictim'], errors='coerce')
process_data['feellikedying'] =pd.to_numeric(process_data['feellikedying'], errors='coerce')
process_data['wannasuicide'] =pd.to_numeric(process_data['wannasuicide'], errors='coerce')
process_data['settlingdown'] =pd.to_numeric(process_data['settlingdown'], errors='coerce')
process_data['ocd'] =pd.to_numeric(process_data['ocd'], errors='coerce')
process_data['SEX'] =pd.to_numeric(process_data['SEX'], errors='coerce')
process_data['MAJORDEPLIFE'] =pd.to_numeric(process_data['MAJORDEPLIFE'], errors='coerce')
process_data['debtridden'] =pd.to_numeric(process_data['debtridden'], errors='coerce')

process_data = process_data.dropna()
#recoding 'drinkingfreq' response variable to be 1 for frequent and 0 for not frequent
"""
def freqbin (row):
   if row in range(1,4):
      return 1
   else:
      return 0
  
  
process_data['drinkingfreq'] = process_data['drinkingfreq'].apply (lambda row: freqbin (row))



# explanatory variables
print('Variable data check')
print('--------------------')
print(process_data['settlingdown'].value_counts(normalize=True).reset_index())
print(process_data['ocd'].value_counts(normalize=True).reset_index())
print(process_data['unpretty'].value_counts(normalize=True).reset_index())
print(process_data['suicidal'].value_counts(normalize=True).reset_index())
print(process_data['SEX'].value_counts(normalize=True).reset_index())
print(process_data['MAJORDEPLIFE'].value_counts(normalize=True).reset_index())
print(process_data['dependence'].value_counts(normalize=True).reset_index())
print(process_data['needy'].value_counts(normalize=True).reset_index())
print(process_data['wannasuicide'].value_counts(normalize=True).reset_index())
print(process_data['feellikedying'].value_counts(normalize=True).reset_index())
print(process_data['anxiety'].value_counts(normalize=True).reset_index())
print(process_data['hoarding'].value_counts(normalize=True).reset_index())
print(process_data['crimevictim'].value_counts(normalize=True).reset_index())
print(process_data['peoplepleasing'].value_counts(normalize=True).reset_index())
print(process_data['fearlonliness'].value_counts(normalize=True).reset_index())
print(process_data['drinkingfreq'].value_counts(normalize=True).reset_index())
print(process_data['debtridden'].value_counts(normalize=True).reset_index())

"""

predictors=process_data[['ocd','needy','unpretty','peoplepleasing','drinkingfreq','debtridden','suicidal','SEX','wannasuicide','fearlonliness','settlingdown','feellikedying','hoarding','dependence','anxiety']]

targets=process_data[['MAJORDEPLIFE']]

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.4)

#print dataset sizes
print(' ')
print(' ')
print('Training set - predictors')
print('--------------------------')
print(pred_train.shape)
print('Test set - predictors')
print('---------------------')
print(pred_test.shape)
print('Target set - predictors')
print('------------------------')
print(tar_train.shape)
print('Target set - test')
print('------------------------')
print(tar_test.shape)

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

print(' ')
print(' ')
print('Confusion Matrix')
print('-----------------')
print(sklearn.metrics.confusion_matrix(predictions,tar_test))

print(' ')
print(' ')
print('Accuracy score')
print('--------------')
print(sklearn.metrics.accuracy_score(tar_test, predictions))
print(' ')
print(' ')

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute


importances=model.feature_importances_
feature_names = predictors.columns # e.g. ['A', 'B', 'C', 'D', 'E']

indices = np.argsort(importances)[::-1]
print("Feature ranking:")
print('-------------------')

for f in range(pred_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
   
#f, ax = plt.subplots(figsize=(5, 6))
plt.title("Feature ranking", fontsize = 11)
plt.bar(range(pred_train.shape[1]), importances[indices],
    color="b", 
    align="center")

plt.xticks(range(pred_train.shape[1]), feature_names[indices])

plt.xticks(rotation=90)
plt.xlim([-1, pred_train.shape[1]])
plt.ylabel("feature importance", fontsize = 11)
plt.xlabel("feature names", fontsize = 11)



# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
#print(model.feature_importances_)


"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction


trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)


plt.cla()
plt.plot(trees, accuracy)
"""