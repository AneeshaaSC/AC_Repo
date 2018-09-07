print(' ')
print('Import Libaries ...')
print(' ')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x:'%f'%x)
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb

print(' ')
print('Read Data from Excel File ...')
print(' ')

numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')


z1data=numoru.copy()
z1pat=z1data[['PatientNumMasked']]
z1data.drop(['PatientNumMasked'],axis=1,inplace=True)
z1data.drop(['LabelRU'],axis=1,inplace=True)
z1tar=numoru[['LabelRU']]
 
z2data=numorm.copy()
z2pat=z2data[['PatientNumMasked']]
z2data.drop(['PatientNumMasked'],axis=1,inplace=True)
z2data.drop(['LabelRM'],axis=1,inplace=True)
z2tar=numorm[['LabelRM']]
 
z3data=numorl.copy()
z3pat=z3data[['PatientNumMasked']]
z3data.drop(['PatientNumMasked'],axis=1,inplace=True)
z3data.drop(['LabelRL'],axis=1,inplace=True)
z3tar=numorl[['LabelRL']]
 
z4data=numolu.copy()
z4pat=z4data[['PatientNumMasked']] 
z4data.drop(['PatientNumMasked'],axis=1,inplace=True)
z4data.drop(['LabelLU'],axis=1,inplace=True)
z4tar=numolu[['LabelLU']]

z5data=numolm.copy()
z5pat=z5data[['PatientNumMasked']]
z5data.drop(['PatientNumMasked'],axis=1,inplace=True)
z5data.drop(['LabelLM'],axis=1,inplace=True)
z5tar=numolm[['LabelLM']]

z6data=numoll.copy()
z6pat=z6data[['PatientNumMasked']]
z6data.drop(['PatientNumMasked'],axis=1,inplace=True)
z6data.drop(['LabelLL'],axis=1,inplace=True)
z6tar=numoll[['LabelLL']]
# no duplicates in z6


def centering(dfname):
    for i in dfname.columns:
            dfname[i]=preprocessing.scale(dfname[i].astype('float64'))
    return dfname

print(' ')
print('Normalize Features ...')
print(' ')


z1data=centering(z1data)    
z2data=centering(z2data)    
z3data=centering(z3data)    
z4data=centering(z4data) 
z5data=centering(z5data)    
z6data=centering(z6data) 

from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import ExtraTreesClassifier    
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split
loo = LeaveOneOut()

print(' ')
print('Perform Feature Selection ...')
print(' ')

#selecting features based on training set
def drop_useless_features(predictors,target):
    #split data set into training and test sets
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.3)
    classifier=ExtraTreesClassifier()
    #fit model on training set alone to perform feature selection
    classifier=classifier.fit(pred_train,tar_train)
    importances=classifier.feature_importances_ 
    sfm = SelectFromModel(classifier,prefit=True)
    predictors = sfm.transform(predictors)
    return predictors,importances

z1data,z1imp=drop_useless_features(z1data,z1tar) 
print('No of features selected in Zone-1: ', z1data.shape[1])
z2data,z2imp=drop_useless_features(z2data,z2tar) 
print('No of features selected in Zone-2: ', z2data.shape[1])
z3data,z3imp=drop_useless_features(z3data,z3tar) 
print('No of features selected in Zone-3: ', z3data.shape[1])
z4data,z4imp=drop_useless_features(z4data,z4tar) 
print('No of features selected in Zone-4: ', z4data.shape[1])
z5data,z5imp=drop_useless_features(z5data,z5tar) 
print('No of features selected in Zone-5: ', z5data.shape[1])
z6data,z6imp=drop_useless_features(z6data,z6tar) 
print('No of features selected in Zone-6: ', z6data.shape[1])


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print(' ')
print('Model Building with Extra Trees Classifier ...')
print(' ')
print(' ')


def model_build_etc(dfpred,dftar,dfpat):
    etc = ExtraTreesClassifier()
    pred_train, pred_test, tar_train, tar_test ,tarpat,testpat = train_test_split(dfpred, dftar,dfpat ,test_size=.3,random_state=123)
    #tarpat,testpat  = train_test_split(dfpat, test_size=.3,random_state=123)
    tar_train=tar_train.values
    #pred_train=pred_train.values
    score=[0]*pred_train.shape[0]         
    for train_index, test_index in loo.split(pred_train):
        X_train, X_test = pred_train[train_index], pred_train[test_index]
        y_train= tar_train[train_index]
        y_test = tar_train[test_index]
        etc.fit(X_train, y_train)
        y_test_pred=etc.predict(X_test)
        test_index=np.asscalar(test_index)
        score[test_index]=accuracy_score(y_test_pred,y_test)
    cv_acc_score=np.mean(score)*100
    print('Accuracy on Cross-validation set: ',cv_acc_score,'%')
    print(' ')
    #print('Predict on Test Set ...')
    #print(' ')
    predictions=etc.predict(pred_test)
    test_acc_score=accuracy_score(tar_test, predictions)*100
    print('Accuracy Score on Test Set',test_acc_score,'%')
    #print(testpat.shape)
    #print(testpat.head(5))
    return predictions,testpat

print('Zone -1 ...')
print(' ')
y1_pred,y1pat=model_build_etc(z1data,z1tar,z1pat)
print(' ')
#print('y1pat...')
#print(y1pat.head(5))

y1_pred=pd.DataFrame(y1_pred,columns=['y1pred'])
#print('y1_pred after making it a df...')
#print(y1_pred.head(5))

y1pat.reset_index(inplace=True)
y1_pred.reset_index(inplace=True)
#print('y1pat after assigning index')
#print(y1pat.head(5))

#print('y1_pred after making assigning index...')
#print(y1_pred.head(5))


#y1_pred=pd.merge(y1_pred,y1pat,on='index')
y1_pred['PatientNumMasked']=y1pat['PatientNumMasked']
#y1_pred.drop(['index'],axis=1,inplace=True)
#print('y1_pred after merging on index...')
#print(y1_pred.iloc[[3]])

print('Zone -2 ...')
print(' ')
y2_pred,y2pat=model_build_etc(z2data,z2tar,z2pat)
print(' ')
y2_pred=pd.DataFrame(y2_pred,columns=['y2pred'])
y2pat.reset_index(inplace=True)
y2_pred.reset_index(inplace=True)
y2_pred['PatientNumMasked']=y2pat['PatientNumMasked']

print('Zone -3 ...')
print(' ')
y3_pred,y3pat=model_build_etc(z3data,z3tar,z3pat)
print(' ')
y3_pred=pd.DataFrame(y3_pred,columns=['y3pred'])
y3pat.reset_index(inplace=True)
y3_pred.reset_index(inplace=True)
y3_pred['PatientNumMasked']=y3pat['PatientNumMasked']

print('Zone -4 ...')
print(' ')
y4_pred,y4pat=model_build_etc(z4data,z4tar,z4pat)
print(' ')
y4_pred=pd.DataFrame(y4_pred,columns=['y4pred'])
y4pat.reset_index(inplace=True)
y4_pred.reset_index(inplace=True)
y4_pred['PatientNumMasked']=y4pat['PatientNumMasked']

print('Zone -5 ...')
print(' ')
y5_pred,y5pat=model_build_etc(z5data,z5tar,z5pat)
print(' ')
y5_pred=pd.DataFrame(y5_pred,columns=['y5pred'])
y5pat.reset_index(inplace=True)
y5_pred.reset_index(inplace=True)
y5_pred['PatientNumMasked']=y5pat['PatientNumMasked']

print('Zone -6 ...')
print(' ')
y6_pred,y6pat=model_build_etc(z6data,z6tar,z6pat)
y6_pred=pd.DataFrame(y6_pred,columns=['y6pred'])
y6pat.reset_index(inplace=True)
y6_pred.reset_index(inplace=True)
y6_pred['PatientNumMasked']=y6pat['PatientNumMasked']

print(' ')
print(' ')
y1_pred.drop(['index'],axis=1,inplace=True)
y2_pred.drop(['index'],axis=1,inplace=True)
y3_pred.drop(['index'],axis=1,inplace=True)
y4_pred.drop(['index'],axis=1,inplace=True)
y5_pred.drop(['index'],axis=1,inplace=True)
y6_pred.drop(['index'],axis=1,inplace=True)

#print('Creating ypred ...')
#print(' ')
#print(' ')
ypred=pd.merge(y1_pred,y2_pred, on='PatientNumMasked',how='outer')
ypred=pd.merge(ypred,y3_pred, on='PatientNumMasked',how='outer')
ypred=pd.merge(ypred,y4_pred, on='PatientNumMasked',how='outer')
ypred=pd.merge(ypred,y5_pred, on='PatientNumMasked',how='outer')
ypred=pd.merge(ypred,y6_pred, on='PatientNumMasked',how='outer')
ypred.fillna(0, inplace=True)

#print(ypred.describe())

ypred['Ypred']=ypred[['y1pred','y2pred','y3pred','y4pred','y5pred','y6pred']].max(axis=1)

z1pred=numoru.copy()
z1y=z1pred[['PatientNumMasked','LabelRU']]

z2pred=numorm.copy()
z2y=z2pred[['PatientNumMasked','LabelRM']]

z3pred=numorl.copy()
z3y=z3pred[['PatientNumMasked','LabelRL']]

z4pred=numolu.copy()
z4y=z4pred[['PatientNumMasked','LabelLU']]

z5pred=numolm.copy()
z5y=z5pred[['PatientNumMasked','LabelLM']]

z6pred=numoll.copy()
z6y=z6pred[['PatientNumMasked','LabelLL']]
"""
print(z1y.head(5))
print(z2y.head(5))
print(z3y.head(5))
print(z4y.head(5))
print(z5y.head(5))
print(z6y.head(5))
"""
def make_actual_y():
    pred_train, pred_test, tar_train, y1  = train_test_split(z1pred, z1y, test_size=.3,random_state=123)
    pred_train, pred_test, tar_train, y2  = train_test_split(z2pred, z2y, test_size=.3,random_state=123)
    pred_train, pred_test, tar_train, y3  = train_test_split(z3pred, z3y, test_size=.3,random_state=123)
    pred_train, pred_test, tar_train, y4  = train_test_split(z4pred, z4y, test_size=.3,random_state=123)
    pred_train, pred_test, tar_train, y5  = train_test_split(z5pred, z5y, test_size=.3,random_state=123)
    pred_train, pred_test, tar_train, y6  = train_test_split(z6pred, z6y, test_size=.3,random_state=123)  
    ymain=pd.merge(y1,y2, on='PatientNumMasked',how='outer')
    ymain=pd.merge(ymain,y3, on='PatientNumMasked',how='outer')
    ymain=pd.merge(ymain,y4, on='PatientNumMasked',how='outer')
    ymain=pd.merge(ymain,y5, on='PatientNumMasked',how='outer')
    ymain=pd.merge(ymain,y6, on='PatientNumMasked',how='outer')
    ymain.fillna(0, inplace=True)
    #print(ymain.describe())
    ymain['Y']=ymain[["LabelRU", "LabelRM","LabelRL", "LabelLU","LabelLM", "LabelLL"]].max(axis=1)
    return ymain


#print('Creating ymain ...')
print(' ')
print(' ')
y_main=make_actual_y()

#print(y_main.head(5))

#print(ypred.head(5))

  
a=pd.merge(y_main,ypred, on='PatientNumMasked')


print(ypred.shape[0])
print(y_main.shape[0])


print('Calculating accuracy ...')
print(' ')
print(' ')
final_acc=accuracy_score(y_main[['Y']],ypred[['Ypred']])
print('Final Accuracy score:',final_acc*100)



#print('Creating confusion matrix ...')
#print(' ')
#print(' ')
cm=confusion_matrix(y_main['Y'],ypred['Ypred'])
#cm=model_build_etc(comb,combtar)
print(' ')
print(' ')
print('Confusion Matrix')
print('~~~~~~~~~~~~~~~~~')
print(cm)


True_positives=cm[1][1]
True_negatives=cm[0][0]
False_negatives=cm[1][0]
False_positives=cm[0][1]
print(' ')
print(' ')

print('True Negatives=',True_negatives)
print('False Negatives=',False_negatives)
print('True Positives=',True_positives)
print('False positives=',False_positives)

print(' ')
print(' ')
p=True_positives/(True_positives+False_positives)
r=True_positives/(True_positives+False_negatives)
print('Precision=',p)
print(' ')
print(' ')
print('Recall=',r)
