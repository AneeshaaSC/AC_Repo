import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x:'%f'%x)
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf


# read data from all different tabs in excel 
numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')

comb=pd.read_excel('combined.xlsx')
combcp=comb[['PatientNumMasked','Zone']]
excomb=comb[['PatientNumMasked','Zone','Label']]
comb_copy=comb.copy()
combtar=comb[['Label']]
combcp=pd.DataFrame((combcp))
comb.drop(['PatientNumMasked', 'Label','Zone'], axis=1,inplace=True)

def centering(dfname):
    for i in dfname.columns:
            dfname[i]=preprocessing.scale(dfname[i].astype('float64'))
    return dfname

#print(' ')
#print('Centering Variables ...')
#print(' ')
#comb=centering(comb)

from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

# Perform feature selection   
from sklearn.ensemble import ExtraTreesClassifier    
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.cross_validation import train_test_split

print(' ')
print('Feature Selection ...')
print(' ')

#selecting features based on training set
def drop_useless_features(predictors,target):
    #split data set into training and test sets
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.3,random_state=123)
    classifier=ExtraTreesClassifier()
    #fit model on training set alone to perform feature selection
    classifier=classifier.fit(pred_train,tar_train)
    importances=classifier.feature_importances_ 
    y_pred=classifier.predict(pred_test)
    acc_pre_sel=accuracy_score(y_pred,tar_test)*100
    print('Accuracy before feature selection: ',acc_pre_sel)
    sfm = SelectFromModel(classifier,prefit=True)
    predictors = sfm.transform(predictors)
    #print('Number features selected: ',predictors.shape[1])
    #print('shape of test data before transform',pred_test.shape[1])
    #print('shape of train data before transform',pred_train.shape[1])
    pred_test  = sfm.transform(pred_test)
    pred_train = sfm.transform(pred_train)
    #print('shape of test data after transform',pred_test.shape[1])
    #print('shape of train data after transform',pred_train.shape[1])
    classifier=classifier.fit(pred_train,tar_train)
    y_pred=classifier.predict(pred_test)
    acc_post_sel=accuracy_score(y_pred,tar_test)*100
    print('Accuracy after feature selection: ',acc_post_sel)    
    #print(pred_test.shape[1])
    #importances=classifier.feature_importances_ 
    #after feature selection, evaluate accuracy using loo cross validation
    #acc_post_feature_sel = cross_val_score(classifier, pred_train, tar_train,cv=loo)
    #acc_post_feature_sel=acc_post_feature_sel.mean()*100
    #print('Accuracy before feature selection', acc_post_feature_sel,'%')

    return predictors,importances


comb,comb_imp=drop_useless_features(comb,combtar) 
print(comb.shape)
print(' ')
print(' ')

##########after dropping outliers#########################



"""
from sklearn.ensemble import IsolationForest

# fit the model
def iso(dfname):
    clf = IsolationForest(max_samples=100, random_state=123)
    clf.fit(dfname)
    y_pred_train = clf.predict(dfname)
    #s=[(((y_pred_train!=1) & (y_pred_train!=-1)))]
    #s=pd.DataFrame((s))
    #s.columns=["Outlierornot"]
    #print(s.value_counts())
    num_normal = (y_pred_train == 1).sum()
    num_outliers = (y_pred_train == -1).sum()
    print(' ')
    print(' ')
    print('Number of normal observations: ',num_normal)
    print('Number of normal outliers: ',num_outliers)
    print(' ')
    print(' ')
    y_pred_train=pd.DataFrame((y_pred_train))
    y_pred_train.columns=["Outlierornot"]
    y_pred_train.reset_index(inplace=True)
    return y_pred_train

print(comb.shape[0])
out=iso(comb)
combcp.reset_index(inplace=True)
combcp=pd.merge(combcp, out, on='index')

combcp=combcp[['PatientNumMasked','Outlierornot']]

value_list = [-1]
#Grab DataFrame rows where column has certain values

outlist=(combcp[combcp.Outlierornot.isin(value_list)])
#print(outlist)
#outlist.drop_duplicates(inplace=True)
print(' ')
print(' ')
print(outlist["Outlierornot"].value_counts())
#outlist=outlist.PatientNumMasked.unique()
#print(outlist)
#Outliers.drop_duplicates(inplace=True)

print(' ')
print(' ')
#print('Lung Zone    No. of Outliers')
#print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

comb.reset_index(inplace=True)
excomb.reset_index(inplace=True)
comb1=pd.merge(excomb, comb, on='index')


l=outlist["PatientNumMasked"]
print(comb1.shape[0])
comb1 = comb1[~combcp['PatientNumMasked'].isin(l)]
print(comb1.shape[0])

#print(comb1.describe())
comb1tar=comb1['Label']
comb1.drop(['PatientNumMasked', 'Label','Zone','index'], axis=1,inplace=True)
comb1=centering(comb1)
print(comb.shape)
print(' ')
print(' ')
print('Number of observations without outliers: ',comb1.shape)
print(' ')
print(' ')
"""
#comb1,comb_imp=drop_useless_features(comb1,comb1tar) 



def plot_feature_importances(importances,predictors):
    feature_names = predictors.columns
    indices = np.argsort(importances)[::-1]  
    print("Feature ranking:")
    print('-------------------')
    for f in range(predictors.shape[1]):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
    plt.title("Feature Importances", fontsize = 11)
    plt.bar(range(predictors.shape[1]), importances[indices],
    color="b", 
    align="center")
    plt.xticks(range(predictors.shape[1]), feature_names[indices])
    plt.xticks(rotation=90)
    plt.xlim([-1, predictors.shape[1]])
    plt.y("Feature importance", fontsize = 11)
    plt.x("Feature Names", fontsize = 11)
    
print(' ')
print(' ')
#comb_forplot=comb.copy()
comb_copy.drop(['PatientNumMasked','Zone','Label'],axis=1,inplace=True)
 
  
plot_feature_importances(comb_imp,comb_copy)