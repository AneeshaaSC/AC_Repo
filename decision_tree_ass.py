# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:12:54 2015

@author: aneeshaa
"""


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
#import os
import seaborn as sb
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
import itertools
import warnings
warnings.filterwarnings("ignore")

#os.chdir("C:")

"""
Data Engineering and Analysis
"""
#Load the dataset
nesarc=pd.read_csv('nesarc_pds.csv',low_memory=False)

nesarc = nesarc.dropna()

"""
Modeling and Prediction
"""

# Select variables to prove hypothesis
process_data = nesarc[['S4AQ4A18','S4AQ4A17','S13Q5','S10Q1A14','S10Q1A15','OBCOMDX2','S10Q1A9','S4AQ4A16','GENAXDX12','S10Q1A5','SEX','MAJORDEPLIFE','S10Q1A12','S10Q1A21','S5Q6A9']]


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

# Exclude unknowns from analysis- data management
#process_data['needy']=process_data['needy'].replace(9, np.nan) 
#process_data['fearlonliness']=process_data['fearlonliness'].replace(9, np.nan)
process_data['unpretty']=process_data['unpretty'].replace(9, np.nan)

#process_data['settlingdown']=process_data['settlingdown'].replace('BL', np.nan)
#process_data['settlingdown'] =pd.to_numeric(process_data['settlingdown'], errors='coerce')
#process_data['settlingdown']=process_data['settlingdown'].replace(9, np.nan)

process_data['feellikedying']=process_data['feellikedying'].replace('BL', np.nan)
process_data['feellikedying'] =pd.to_numeric(process_data['feellikedying'], errors='coerce')

process_data['feellikedying']=process_data['feellikedying'].replace(9, np.nan)


process_data['wannasuicide']=process_data['wannasuicide'].replace('BL', np.nan)
process_data['wannasuicide'] =pd.to_numeric(process_data['wannasuicide'], errors='coerce')
process_data['wannasuicide']=process_data['wannasuicide'].replace(9, np.nan)
#process_data['dependence']=process_data['dependence'].replace(9, np.nan)

process_data['suicidal']=process_data['suicidal'].replace('BL', np.nan)
process_data['suicidal'] =pd.to_numeric(process_data['suicidal'], errors='coerce')
process_data['suicidal']=process_data['suicidal'].replace(9, np.nan)
#process_data['anxiety'] =process_data['anxiety'].replace(9, np.nan)
#process_data['peoplepleasing'] =process_data['peoplepleasing'].replace(9, np.nan)
process_data['hoarding'] =process_data['hoarding'].replace(9, np.nan)
#process_data['crimevictim'] =process_data['crimevictim'].replace(99, np.nan)

process_data = process_data.dropna()



# convert variables to numeric
#process_data['needy'] =pd.to_numeric(process_data['needy'], errors='coerce')
process_data['suicidal'] =pd.to_numeric(process_data['suicidal'], errors='coerce')
process_data['unpretty'] =pd.to_numeric(process_data['unpretty'], errors='coerce')
#process_data['anxiety'] =pd.to_numeric(process_data['anxiety'], errors='coerce') 
process_data['SEX'] =pd.to_numeric(process_data['SEX'], errors='coerce') 
process_data['MAJORDEPLIFE'] =pd.to_numeric(process_data['MAJORDEPLIFE'], errors='coerce') 
#process_data['peoplepleasing'] =pd.to_numeric(process_data['peoplepleasing'], errors='coerce') 
#process_data['dependence'] =pd.to_numeric(process_data['dependence'], errors='coerce')
process_data['hoarding'] =pd.to_numeric(process_data['hoarding'], errors='coerce') 
#process_data['crimevictim'] =pd.to_numeric(process_data['crimevictim'], errors='coerce')
process_data['feellikedying'] =pd.to_numeric(process_data['feellikedying'], errors='coerce')
process_data['wannasuicide'] =pd.to_numeric(process_data['wannasuicide'], errors='coerce')
process_data['settlingdown'] =pd.to_numeric(process_data['settlingdown'], errors='coerce')
process_data['ocd'] =pd.to_numeric(process_data['ocd'], errors='coerce')
#process_data['SEX'] =pd.to_numeric(process_data['SEX'], errors='coerce')
#process_data['MAJORDEPLIFE'] =pd.to_numeric(process_data['MAJORDEPLIFE'], errors='coerce')


process_data = process_data.dropna()
#recoding 'SEX' response variable to be 1 for female and 0 for male
"""
def femaleonly (row):
   if row==1:
      return 0
   else:
      return 1
  
  
process_data['SEX'] = process_data['SEX'].apply (lambda row: femaleonly (row))
"""
#process_data = process_data.dropna()
# explanatory variables
print('Variable data check')
print('--------------------')
print(process_data['settlingdown'].value_counts(normalize=True).reset_index())
print(process_data['ocd'].value_counts(normalize=True).reset_index())
print(process_data['unpretty'].value_counts(normalize=True).reset_index())
print(process_data['suicidal'].value_counts(normalize=True).reset_index())
print(process_data['SEX'].value_counts(normalize=True).reset_index())
print(process_data['MAJORDEPLIFE'].value_counts(normalize=True).reset_index())
#print(process_data['dependence'].value_counts(normalize=True).reset_index())
#print(process_data['needy'].value_counts(normalize=True).reset_index())
print(process_data['wannasuicide'].value_counts(normalize=True).reset_index())
print(process_data['feellikedying'].value_counts(normalize=True).reset_index())
print(process_data['anxiety'].value_counts(normalize=True).reset_index())
print(process_data['hoarding'].value_counts(normalize=True).reset_index())
#print(process_data['crimevictim'].value_counts(normalize=True).reset_index())
#print(process_data['peoplepleasing'].value_counts(normalize=True).reset_index())
#print(process_data['fearlonliness'].value_counts(normalize=True).reset_index())



predictors = process_data[['SEX','feellikedying','unpretty','wannasuicide','ocd','hoarding']]

targets = process_data[['MAJORDEPLIFE']]

#print(predictors.describe())
#ax = sb.countplot(x="MAJORDEPLIFE", hue="SEX", data=l)

#Split into training and testing sets
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

#Build model on training data
myclassifier=DecisionTreeClassifier()
myclassifier=myclassifier.fit(pred_train,tar_train)

predictions=myclassifier.predict(pred_test)
print(' ')
print(' ')
print('Test Set Confusion Matrix')
print('--------------------------')
print(sklearn.metrics.confusion_matrix(tar_test,predictions))

print(' ')
print(' ')
print('Accuracy score')
print('--------------------------')
print(sklearn.metrics.accuracy_score(tar_test, predictions))

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
#from IPython.display import Image
out = StringIO()

with open("classifier.txt", "w") as f:
    f = tree.export_graphviz(myclassifier, out_file=f)

#tree.export_graphviz(classifier, out_file=out)


#import pydotplus
#graph=pydotplus.graph_from_dot_data(out.getvalue())
#Image(graph.create_png())

"""

def plot_confusion_matrix(cm, classes,
			  normalize=False,
			  title='Confusion matrix',
			  cmap=plt.cm.Blues):

#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
		horizontalalignment="center",
		color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = sklearn.metrics.confusion_matrix(tar_test, predictions)
np.set_printoptions(precision=2)
classes=['Male','Female']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,  classes ,
		      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes, normalize=True,
		      title='Normalized confusion matrix')

plt.show()


"""
