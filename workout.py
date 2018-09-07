# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:38:33 2017

@author: 212458792
"""
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
numorm_copy=numorm.copy()
#check dimensions
print(' ')
print(' ')
print('Number of samples with RightUpper data(Zone-1):',numoru.shape[0]) 
print('Number of samples with RightMiddle data(Zone-2):',numorm.shape[0])
print('Number of samples with Rightlower data(Zone-3):',numorl.shape[0])
print('Number of samples with LeftUpper data(Zone-4):',numolu.shape[0])
print('Number of samples with LeftMiddle data(Zone-5):',numolm.shape[0])
print('Number of samples with LeftLower data(Zone-6):',numoll.shape[0])
print(' ')
print(' ')


# Merge PatientNumMased and s of the 6 zones into one dataframe: ymain
ymain=numorm[['PatientNumMasked','LabelRM']]
ymain.rename(columns={'LabelRM':'y2'},inplace=True)
ymain=pd.merge(numoru[['LabelRU','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'LabelRU':'y1'},inplace=True)
ymain=pd.merge(numorl[['LabelRL','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'LabelRL':'y3'},inplace=True)
ymain=pd.merge(numolu[['LabelLU','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'LabelLU':'y4'},inplace=True)
ymain=pd.merge(numolm[['LabelLM','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'LabelLM':'y5'},inplace=True)
ymain=pd.merge(numoll[['LabelLL','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'LabelLL':'y6'},inplace=True)

# Replace Nans with 0 or 1 based on labels for other zones in ymain
ymain.fillna(0, inplace=True)

# If any one of the zones are labeled '1' or abnormal, Y will be 1
ymain['Y']=ymain[["y1", "y2","y3", "y4","y5", "y6"]].max(axis=1)
#print(ymain['y1'].value_counts(dropna=False))
#print(ymain['y2'].value_counts(dropna=False))
#print(ymain['y3'].value_counts(dropna=False))
#print(ymain['y4'].value_counts(dropna=False))
#print(ymain['y5'].value_counts(dropna=False))
#print(ymain['y6'].value_counts(dropna=False))
#separate predictors and target variables into different dataframes for all 6 zones
z1data=numoru.copy()
#check for duplicates, duplicate PatientNumMasked vlaues will be True
#z1dups=z1data.duplicated(['PatientNumMasked'], keep=False)
# print how many duplicates were found
#print('z1 duplicates counts: ',z1dups.groupby(z1dups).size())
# drop identifier and target variable
z1data.drop(['PatientNumMasked'],axis=1,inplace=True)
z1data.drop(['LabelRU'],axis=1,inplace=True)
z1tar=numoru['LabelRU']
# no duplicates in z1
z2data=numorm.copy()
#z2dups=z2data.duplicated(['PatientNumMasked'], keep=False)
#print('z2 duplicates counts: ',z2dups.groupby(z2dups).size())
z2data.drop(['PatientNumMasked'],axis=1,inplace=True)
z2data.drop(['LabelRM'],axis=1,inplace=True)
z2tar=numorm['LabelRM']
# no duplicates in z2
z3data=numorl.copy()
#z3dups=z3data.duplicated(['PatientNumMasked'], keep=False)
#print('z3 duplicates counts: ',z3dups.groupby(z3dups).size())
z3data.drop(['PatientNumMasked'],axis=1,inplace=True)
z3data.drop(['LabelRL'],axis=1,inplace=True)
z3tar=numorl['LabelRL']
# no duplicates in z3
z4data=numolu.copy()
#z4dups=z4data.duplicated(['PatientNumMasked'], keep=False)
#print('z4 duplicates counts: ',z4dups.groupby(z4dups).size())
z4data.drop(['PatientNumMasked'],axis=1,inplace=True)
z4data.drop(['LabelLU'],axis=1,inplace=True)
z4tar=numolu['LabelLU']
# no duplicates in z4
z5data=numolm.copy()
#z5dups=z5data.duplicated(['PatientNumMasked'], keep=False)
#print('z5 duplicates counts: ',z5dups.groupby(z5dups).size())
z5data.drop(['PatientNumMasked'],axis=1,inplace=True)
z5data.drop(['LabelLM'],axis=1,inplace=True)
z5tar=numolm['LabelLM']
# no duplicates in z5
z6data=numoll.copy()
#z6dups=z6data.duplicated(['PatientNumMasked'], keep=False)
#print('z6 duplicates counts: ',z6dups.groupby(z6dups).size())
z6data.drop(['PatientNumMasked'],axis=1,inplace=True)
z6data.drop(['LabelLL'],axis=1,inplace=True)
z6tar=numoll['LabelLL']
# no duplicates in z6


# standardize clustering variables to have mean=0 and sd=1
def centering(dfname):
    for i in dfname.columns:
            dfname[i]=preprocessing.scale(dfname[i].astype('float64'))
    return dfname

print(' ')
print('Centering Variables ...')
print(' ')
z1data=centering(z1data)    
z2data=centering(z2data)    
z3data=centering(z3data)    
z4data=centering(z4data) 
z5data=centering(z5data)    
z6data=centering(z6data) 

# to perform leave one out cross validation
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
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.2,random_state=123)
    classifier=ExtraTreesClassifier()
    #fit model on training set alone to perform feature selection
    classifier=classifier.fit(pred_train,tar_train)
    sfm = SelectFromModel(classifier,prefit=True)
    predictors = sfm.transform(predictors)
    print('Number features selected: ',predictors.shape[1])
    importances=classifier.feature_importances_ 
    #after feature selection, evaluate accuracy using loo cross validation
    #acc_post_feature_sel = cross_val_score(classifier, pred_train, tar_train,cv=loo)
    #acc_post_feature_sel=acc_post_feature_sel.mean()*100
    #print('Accuracy before feature selection', acc_post_feature_sel,'%')
    return predictors,importances

print('Zone-1')
print('~~~~~~~')
z1data,z1imp=drop_useless_features(z1data,z1tar) 
print(z1data.shape)
print(' ')
print(' ')

print('Zone-2')
print('~~~~~~~')
z2data,z2imp=drop_useless_features(z2data,z2tar) 
print(z2data.shape)
print(' ')
print(' ')

print('Zone-3')
print('~~~~~~~')
z3data,z3imp=drop_useless_features(z3data,z3tar) 
print(z3data.shape)
print(' ')
print(' ')

print('Zone-4')
print('~~~~~~~')
z4data,z4imp=drop_useless_features(z4data,z4tar) 
print(z4data.shape)
print(' ')
print(' ')

print('Zone-5')
print('~~~~~~~')
z5data,z5imp=drop_useless_features(z5data,z5tar) 
print(z5data.shape)
print(' ')
print(' ')

print('Zone-6')
print('~~~~~~~')
z6data,z6imp=drop_useless_features(z6data,z6tar) 
print(z6data.shape)
print(' ')
print(' ')

print(' ')
print('Feature Selection complete!')
print(' ')

print(' ')
print('Plot of feature importances for zone 2')
print(' ')

def plot_feature_importances(importances,predictors):
    feature_names = predictors.columns
    indices = np.argsort(importances)[::-1]  
    print("Feature ranking:")
    print('-------------------')
    for f in range(predictors.shape[1]):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
    #plt.title("Feature Importances", fontsize = 11)
    #plt.bar(range(predictors.shape[1]), importances[indices],
    #color="b", 
    #align="center")
    #plt.xticks(range(predictors.shape[1]), feature_names[indices])
    #plt.xticks(rotation=90)
    #plt.xlim([-1, predictors.shape[1]])
    #plt.y("Feature importance", fontsize = 11)
    #plt.x("Feature Names", fontsize = 11)
    
print(' ')
print(' ')
z2data_forplot=numorm.copy()
z2data_forplot.drop(['PatientNumMasked'],axis=1,inplace=True)
z2data_forplot.drop(['LabelRM'],axis=1,inplace=True)
plot_feature_importances(z2imp,z2data_forplot)

print(' ')
print(' ')
#z1data_forplot=numoru.copy()
#z1data_forplot.drop(['PatientNumMasked'],axis=1,inplace=True)
#z1data_forplot.drop(['LabelRU'],axis=1,inplace=True)
#plot_feature_importances(z1imp,z1data_forplot)
"""
print(' ')
print(' ')
z4data_forplot=numolu.copy()
z4data_forplot.drop(['PatientNumMasked'],axis=1,inplace=True)
z4data_forplot.drop(['LabelLU'],axis=1,inplace=True)
plot_feature_importances(z4imp,z4data_forplot)

numorm['Hist_2_150_2_Entropy_cats']=pd.qcut(numorm.Hist_2_150_2_Entropy,4,labels=["Very Low","Low","Average","High"])
print(' ')
print(' ')
numorm["Hist_2_150_2_Entropy_cats"] = numorm["Hist_2_150_2_Entropy_cats"].astype('category')
print('Hist_2_150_2_Entrop category counts')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
cnt = numorm['Hist_2_150_2_Entropy_cats'].value_counts(sort=False, dropna=True)
print(cnt)

print(' ')
print(' ')
numorm['LabelRM'] = numorm['LabelRM'].convert_objects(convert_numeric=True)

sb.factorplot(x="Hist_2_150_2_Entropy_cats", y="LabelRM", data=numorm, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'])
plt.xlabel('Hist_2_150_2_Entropy Categories')
plt.ylabel('Proportion of Patients with Pneumoconiosis')

#############################################

numorm['Hist_2_30_2_Entropy_cats']=pd.qcut(numorm.Hist_2_30_2_Entropy,4,labels=["Very Low","Low","Average","High"])
print(' ')
print(' ')
numorm["Hist_2_30_2_Entropy_cats"] = numorm["Hist_2_30_2_Entropy_cats"].astype('category')
print('Hist_2_30_2_Entropy category counts')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
cnt = numorm['Hist_2_30_2_Entropy_cats'].value_counts(sort=False, dropna=True)
print(cnt)

print(' ')
print(' ')
numorm['LabelRM'] = numorm['LabelRM'].convert_objects(convert_numeric=True)

sb.factorplot(x="Hist_2_30_2_Entropy_cats", y="LabelRM", data=numorm, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'])
plt.xlabel('Hist_2_30_2_Entropy Categories')
plt.ylabel('Proportion of Patients with Pneumoconiosis')




numorm_copy['LabelRM'] = numorm_copy['LabelRM'].convert_objects(convert_numeric=True)
"""
"""
numorm_copy=numorm.copy()
numorm_copy.drop(['PatientNumMasked'],axis=1,inplace=True)
numorm_copy.drop(['LabelRM'],axis=1,inplace=True)
numorm_copy=centering(numorm_copy)
numorm_copy['LabelRM']=numorm['LabelRM']
numorm_copy['PatientNumMasked']=numorm['LabelRM']

#reg1=smf.logit(formula='LabelRM ~ Hist_1_180_2_StdDev+CoMatrix_Deg135_Inertia+Hist_2_150_2_Entropy + Hist_1_180_2_StdDev + Hist_2_90_1_Kurtosis + CoMatrix_Deg135_Inertia  + CoMatrix_Deg45_Local_Homogeneity + Hist_2_60_2_Skewness +  CoMatrix_Deg135_Local_Homogeneity',data=numorm).fit()


reg2=smf.logit(formula='LabelRM ~ CoMatrix_Deg135_Correlation+CoMatrix_Deg135_Inertia+CoMatrix_Deg135_Local_Homogeneity+CoMatrix_Deg45_Local_Homogeneity+CoMatrix_Deg90_Local_Homogeneity+Hist_0_0_0_Entropy+Hist_0_0_0_Kurtosis+Hist_0_0_0_Mean+Hist_0_0_0_Skewness+Hist_1_120_2_Mean+Hist_1_135_2_Entropy+Hist_1_135_2_Mean+Hist_1_150_1_Skewness+Hist_1_180_2_Mean+Hist_1_180_2_Skewness+Hist_1_180_2_StdDev+Hist_1_30_2_Mean+Hist_1_90_2_Skewness+Hist_2_135_1_Entropy+Hist_2_150_2_Entropy+Hist_2_150_2_Kurtosis+Hist_2_150_2_Mean+Hist_2_150_2_Skewness+Hist_2_180_1_Skewness+Hist_2_180_2_Entropy+Hist_2_180_2_Kurtosis+Hist_2_180_2_Mean+Hist_2_180_2_Skewness+Hist_2_30_2_Entropy+Hist_2_30_2_Mean+Hist_2_45_1_Entropy+Hist_2_60_1_Skewness+Hist_2_60_2_Kurtosis+Hist_2_60_2_Skewness+Hist_2_90_1_Kurtosis+Hist_2_90_1_Skewness+Hist_2_90_2_Kurtosis+Hist_2_90_2_Mean+Hist_2_90_2_Skewness',data=numorm_copy).fit()


print(' ')
print(' ')
print(reg2.summary())
print(' ')
print(' ')



params = reg2.params
conf = reg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

testdf=numorm[['Hist_1_180_2_StdDev','CoMatrix_Deg135_Inertia','Hist_2_150_2_Entropy','Hist_2_90_1_Kurtosis','CoMatrix_Deg45_Local_Homogeneity','Hist_2_60_2_Skewness','CoMatrix_Deg135_Local_Homogeneity']]


def final_pred(predictors,target):
    classifier=ExtraTreesClassifier()
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.4,random_state=123)
    cv_score=cross_val_score(classifier,pred_train,tar_train,cv=loo)
    tar_pred=cross_val_predict(classifier,pred_test,tar_test)   
    #for train_index, test_index in loo.split(predictors):
    #    X_train, X_test = predictors.iloc[train_index], predictors.iloc[test_index]
    #    y_train= target[train_index]
    #    classifier=classifier.fit(X_train,y_train)    
    score=accuracy_score(tar_test,tar_pred)*100
    return tar_pred,score
"""
# Load libraries
 
#from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def model_build(predictors,target):
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=2)))
    models.append(('CART', ExtraTreesClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    print(predictors.shape)
    print(target.shape)
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.3,random_state=123)
    for name, model in models:
    	cv_results = model_selection.cross_val_score(model, pred_train, tar_train, cv=loo)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
# Compare Algorithms
    fig = plt.figure()
    #fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    #plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.lineplot(cv_results.mean())
     
    plt.title('Model Comparison')
    plt.show()
    """
    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(pred_train, tar_train)
    predictions = knn.predict(pred_test)
    print(accuracy_score(tar_test, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    """
model_build(z2data,z2tar)

        
"""
print(' ')
print('Zone level prediction begins')
print(' ')
 
from sklearn.linear_model import LogisticRegression



def get_y_pred(predictors,target):
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.4,random_state=123)
    model = LogisticRegression()
    cv_score=cross_val_score(model,pred_train,tar_train,cv=loo)
    pred_y = cross_val_predict(model, pred_test, tar_test)
    acc=accuracy_score(pred_y,tar_test)*100
    #acc = model_selection.cross_val_score(model, dfname, df_tar, cv=loo)
    #matrix = confusion_matrix(pred_y, df_tar)
    return pred_y,acc

testdfprd,testdfacclr=get_y_pred(testdf,z2tar)
testdfprd,testdfaccert=final_pred(testdf,z2tar)
print('lr accuracy',testdfacclr)
print('ert accuracy',testdfaccert)
"""
"""
y2_pred,y2_acc=get_y_pred(z2data, z2tar)
y1_pred,y1_acc=get_y_pred(z1data, z1tar)
y3_pred,y3_acc=get_y_pred(z3data, z3tar)
y4_pred,y4_acc=get_y_pred(z4data, z4tar)
y5_pred,y5_acc=get_y_pred(z5data, z5tar)
y6_pred,y6_acc=get_y_pred(z6data, z6tar)

print(' ')
print('convert predictors to dataframe!')
print(' ')

z2data=pd.DataFrame(z2data)
z1data=pd.DataFrame(z1data)
z3data=pd.DataFrame(z3data)
z4data=pd.DataFrame(z4data)
z5data=pd.DataFrame(z5data)
z6data=pd.DataFrame(z6data)
print(' ')
print('call function!')
print(' ')

y2_pred_2,y2_acc_2=final_pred(z2data, z2tar)
y1_pred_2,y1_acc_2=final_pred(z1data, z1tar)
y3_pred_2,y3_acc_2=final_pred(z3data, z3tar)
y4_pred_2,y4_acc_2=final_pred(z4data, z4tar)
y5_pred_2,y5_acc_2=final_pred(z5data, z5tar)
y6_pred_2,y6_acc_2=final_pred(z6data, z6tar)

print(' ')
print(' Prediction for the 6 zones done!')
print(' ')

print('Accuracy with Logistic Regression:')
print("Accuracy for y2: %.3f%% (%.3f%%)" % (y2_acc.mean()*100.0, y2_acc.std()*100.0))
print("Accuracy for y3: %.3f%% (%.3f%%)" % (y3_acc.mean()*100.0, y3_acc.std()*100.0))
print("Accuracy for y1: %.3f%% (%.3f%%)" % (y1_acc.mean()*100.0, y1_acc.std()*100.0))
print("Accuracy for y4: %.3f%% (%.3f%%)" % (y4_acc.mean()*100.0, y4_acc.std()*100.0))
print("Accuracy for y5: %.3f%% (%.3f%%)" % (y5_acc.mean()*100.0, y5_acc.std()*100.0))
print("Accuracy for y6: %.3f%% (%.3f%%)" % (y6_acc.mean()*100.0, y6_acc.std()*100.0))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(' ')
print(' ')
print('Accuracy with Extremely Randomized Trees:')
print("Accuracy for y2: %.3f%% " % (y2_acc_2))
print("Accuracy for y3: %.3f%% " % (y3_acc_2))
print("Accuracy for y1: %.3f%% " % (y1_acc_2))
print("Accuracy for y4: %.3f%% " % (y4_acc_2))
print("Accuracy for y5: %.3f%% " % (y5_acc_2))
print("Accuracy for y6: %.3f%% " % (y6_acc_2))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(' ')
print(' ')
y2_pred=pd.DataFrame(y2_pred,columns = ["y2"])
y1_pred=pd.DataFrame(y1_pred,columns = ["y1"])
y3_pred=pd.DataFrame(y3_pred,columns = ["y3"])
y4_pred=pd.DataFrame(y4_pred,columns = ["y4"])
y5_pred=pd.DataFrame(y5_pred,columns = ["y5"])
y6_pred=pd.DataFrame(y6_pred,columns = ["y6"])

y2_pred_2=pd.DataFrame(y2_pred_2,columns = ["y2"])
y1_pred_2=pd.DataFrame(y1_pred_2,columns = ["y1"])
y3_pred_2=pd.DataFrame(y3_pred_2,columns = ["y3"])
y4_pred_2=pd.DataFrame(y4_pred_2,columns = ["y4"])
y5_pred_2=pd.DataFrame(y5_pred_2,columns = ["y5"])
y6_pred_2=pd.DataFrame(y6_pred_2,columns = ["y6"])

y2_pred.reset_index(inplace=True)
y1_pred.reset_index(inplace=True)
y3_pred.reset_index(inplace=True)
y4_pred.reset_index(inplace=True)
y5_pred.reset_index(inplace=True)
y6_pred.reset_index(inplace=True)

y2_pred_2.reset_index(inplace=True)
y1_pred_2.reset_index(inplace=True)
y3_pred_2.reset_index(inplace=True)
y4_pred_2.reset_index(inplace=True)
y5_pred_2.reset_index(inplace=True)
y6_pred_2.reset_index(inplace=True)

y2_patid=numorm['PatientNumMasked']
y2_patid=pd.DataFrame(y2_patid)
y1_patid=numoru['PatientNumMasked']
y1_patid=pd.DataFrame(y1_patid)
y3_patid=numorl['PatientNumMasked']
y3_patid=pd.DataFrame(y3_patid)
y4_patid=numolu['PatientNumMasked']
y4_patid=pd.DataFrame(y4_patid)
y5_patid=numolm['PatientNumMasked']
y5_patid=pd.DataFrame(y5_patid)
y6_patid=numoll['PatientNumMasked']
y6_patid=pd.DataFrame(y6_patid)



y2_patid.reset_index(inplace=True)
y1_patid.reset_index(inplace=True)
y3_patid.reset_index(inplace=True)
y4_patid.reset_index(inplace=True)
y5_patid.reset_index(inplace=True)
y6_patid.reset_index(inplace=True)

y1_pred=pd.merge(y1_pred, y1_patid, on='index')

y2_pred=pd.merge(y2_pred, y2_patid, on='index')

y3_pred=pd.merge(y3_pred, y3_patid, on='index')

y4_pred=pd.merge(y4_pred, y4_patid, on='index')

y5_pred=pd.merge(y5_pred, y5_patid, on='index')

y6_pred=pd.merge(y6_pred, y6_patid, on='index')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

y1_pred_2=pd.merge(y1_pred_2, y1_patid, on='index')

y2_pred_2=pd.merge(y2_pred_2, y2_patid, on='index')

y3_pred_2=pd.merge(y3_pred_2, y3_patid, on='index')

y4_pred_2=pd.merge(y4_pred_2, y4_patid, on='index')

y5_pred_2=pd.merge(y5_pred_2, y5_patid, on='index')

y6_pred_2=pd.merge(y6_pred_2, y6_patid, on='index')



pred_y=pd.merge(y1_pred, y2_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y3_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y4_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y5_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y6_pred, on='PatientNumMasked',how='outer')
pred_y['Ypred']=pred_y[["y1", "y2","y3", "y4","y5", "y6"]].max(axis=1)

pred_y.fillna(0, inplace=True)

pred_y_2=pd.merge(y1_pred_2, y2_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y3_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y4_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y5_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y6_pred_2, on='PatientNumMasked',how='outer')
pred_y_2['Ypred']=pred_y_2[["y1", "y2","y3", "y4","y5", "y6"]].max(axis=1)

pred_y_2.fillna(0, inplace=True)

from sklearn.metrics import confusion_matrix
ymatrix = confusion_matrix(pred_y['Ypred'], ymain['Y'])
print('mat1')
print(ymatrix)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

ymatrix2 = confusion_matrix(pred_y_2['Ypred'], ymain['Y'])
print('mat2')
print(ymatrix2)


print('y1 shape:',y1_pred.shape) 
print('y2 shape:',y2_pred.shape)
print('y3 shape:',y3_pred.shape)
print('y4 shape:',y4_pred.shape)
print('y5 shape:',y5_pred.shape)
print('y6 shape:',y6_pred.shape)

print(' ')

temp2=pd.merge(ymain, pred_y_2, on='PatientNumMasked')
cp2=0
temp2['Ypred']=temp2['Ypred'].convert_objects(convert_numeric=True)

for i in range(len(temp2)):
    if temp2['Ypred'].iloc[i]==ymain['Y'].iloc[i]:
        cp2=cp2+1

acc2=(cp2/len(ymain))*100
print('Final Accuracy from ERT:', acc2)

from sklearn.metrics import precision_recall_fscore_support as score
precision2, recall2, fscore2, support2 = score(ymain['Y'], pred_y_2['Ypred'])
print('precision1: {}'.format(precision2))
print('recall1: {}'.format(recall2))


temp=pd.merge(ymain, pred_y, on='PatientNumMasked')
cp=0
temp['Ypred']=temp['Ypred'].convert_objects(convert_numeric=True)


for i in range(len(ymain)):
    if temp['Ypred'].iloc[i]==ymain['Y'].iloc[i]:
        cp=cp+1

acc=(cp/len(temp))*100
print('Final Accuracy FROM LR:', acc)


precision, recall, fscore, support = score(ymain['Y'], pred_y['Ypred'])
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))

"""