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

#separate predictors and target variables into different dataframes for all 6 zones
z1data=numoru.copy()
 
z1data.drop(['PatientNumMasked'],axis=1,inplace=True)
z1data.drop(['LabelRU'],axis=1,inplace=True)
z1tar=numoru['LabelRU']
# no duplicates in z1
z2data=numorm.copy()

z2data.drop(['PatientNumMasked'],axis=1,inplace=True)
z2data.drop(['LabelRM'],axis=1,inplace=True)
z2tar=numorm['LabelRM']
# no duplicates in z2
z3data=numorl.copy()
 
z3data.drop(['PatientNumMasked'],axis=1,inplace=True)
z3data.drop(['LabelRL'],axis=1,inplace=True)
z3tar=numorl['LabelRL']
# no duplicates in z3
z4data=numolu.copy()
 
z4data.drop(['PatientNumMasked'],axis=1,inplace=True)
z4data.drop(['LabelLU'],axis=1,inplace=True)
z4tar=numolu['LabelLU']
# no duplicates in z4
z5data=numolm.copy()


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
from sklearn.cluster import KMeans

import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
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

from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

"""
def find_k(dfpred,dftar):
    clus_train, clus_test, tar_train, tar_test  = train_test_split(dfpred, dftar, test_size=.3,random_state=123)
    for i in clusters:
        model=KMeans(n_clusters=i)
        model.fit(clus_train)
        meandist.append(sum(np.min(cdist(clus_train,model.cluster_centers_,'euclidean'),axis=1))/clus_train.shape[0])
    plt.plot(clusters,meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')

find_k(z6data,z6tar)
"""
from sklearn.decomposition import PCA

def plot_clusters(dfpred,dftar):
    clus_train, clus_test, tar_train, tar_test  = train_test_split(dfpred, dftar, test_size=.3,random_state=123)
    model=KMeans(n_clusters=2)
    model.fit(clus_train)
    pca_3=PCA(3)
    pca_matrix=pca_3.fit_transform(clus_train)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(xs=pca_matrix[:,0], ys=pca_matrix[:,1],zs=pca_matrix[:,2],zdir='z',s = 10 , c=model.labels_, depthshade=False)              
    ax.view_init(45, 45)              
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    ax.set_zlabel('Canonical variable 3')
    plt.title('3D Scatterplot of Canonical Variables for 2 Clusters')
    plt.tight_layout()
    plt.show()
    #clusassign=model.predict(clus_train)
    #pca_2=PCA(2)
    #pca_matrix=pca_2.fit_transform(clus_train)
    #plt.scatter(x=pca_matrix[:,0],y=pca_matrix[:,1],c=model.labels_,)
    #plt.xlabel('Canonical variable 1')
    #plt.ylabel('Canonical variable 2')
    #plt.title('Scatterplot of Canonical Variables for 2 Clusters')
    #plt.show()


plot_clusters(z5data,z5tar)
"""
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(xs=pca_matrix[:,0], ys=pca_matrix[:,1],zs=pca_matrix[:,2],zdir='z',
              s = 10 , c=model2.labels_, depthshade=False)              
ax.view_init(45, 45)              
plt.xlabel('Canonicl variable 1')
plt.ylabel('Canonical variable 2')
ax.set_zlabel('Canonical variable 3')
plt.title('3D Scatterplot of Canonical Variables for 2 Clusters')
plt.tight_layout()
plt.show()
#Axes3D.scatter(xs, ys, zs=0, zdir='z’, s=20, c='b’, depthshade=True, *args, **kwargs)
"""