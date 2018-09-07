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
comb=pd.read_excel('combined.xlsx')
comb_copy=comb.copy()
combtar=comb[['Label']]
#combcp=pd.DataFrame((combcp))
comb.drop(['PatientNumMasked', 'Label','Zone'], axis=1,inplace=True)


def centering(dfname):
    for i in dfname.columns:
            dfname[i]=preprocessing.scale(dfname[i].astype('float64'))
    return dfname

print(' ')
print('Normalize Features ...')
print(' ')

comb=centering(comb)


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


comb,comb_imp=drop_useless_features(comb,combtar) 

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

def model_build(predictors,target):
    models = []
    models.append(('LR', LogisticRegression()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=2)))
    models.append(('ETC', ExtraTreesClassifier()))
    #models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    #print(predictors.shape)
    #print(target.shape)
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.3,random_state=123)
    for name, model in models:
    	cv_results = model_selection.cross_val_score(model, pred_train, tar_train, cv=loo)
    	results.append(cv_results.mean()*100)
    	names.append(name)
    	msg = "%s: %f%%" % (name, cv_results.mean()*100 )
    	print(msg)


#model_build(comb,combtar)


def model_build_etc(dfpred,dftar):
    etc = ExtraTreesClassifier()
    pred_train, pred_test, tar_train, tar_test  = train_test_split(dfpred, dftar, test_size=.3)
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
    print(' ')
    print('Predict on Test Set ...')
    print(' ')
    print(' ')
    predictions=etc.predict(pred_test)
    test_acc_score=accuracy_score(tar_test, predictions)*100
    print('Accuracy Score on Test Set',test_acc_score,'%')
    matrix=confusion_matrix(tar_test,predictions)
    return matrix


cm=model_build_etc(comb,combtar)
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

