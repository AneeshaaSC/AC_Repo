# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:38:12 2017

@author: 212458792
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D

wholesale=pd.read_csv('Wholesale customers data.csv',low_memory=False)

wholesale=wholesale.dropna()

clustervars=wholesale[['Region','Fresh', 'Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]

targets=wholesale[['Channel']] 

# standardize clustering variables to have mean=0 and sd=1
for i in clustervars.columns:
    clustervars[i]=preprocessing.scale(clustervars[i].astype('float64'))

#print(clustervars.describe())
# split data set into training and test sets
#clus_train,clus_test=train_test_split(clustervars,test_size=0.3,random_state=123)

clus_train, clus_test, tar_train, tar_test  = train_test_split(clustervars, targets, test_size=.3)

# k means cluster analysis for 1-9 clusters
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for i in clusters:
    model=KMeans(n_clusters=i)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train,model.cluster_centers_,'euclidean'),axis=1))/clus_train.shape[0])
    
#print(clusassign)
"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
"""
plt.plot(clusters,meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
"""

model2=KMeans(n_clusters=2)
model2.fit(clus_train)
clusassign=model2.predict(clus_train)

from sklearn.decomposition import PCA
pca_2=PCA(3)
pca_matrix=pca_2.fit_transform(clus_train)


plt.scatter(x=pca_matrix[:,0],y=pca_matrix[:,1],c=model2.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()
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

"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(inplace=True)
tar_train.reset_index(inplace=True)
#print(clus_train.head(5))
#print(tar_train.head(5))
full=pd.merge(clus_train, tar_train, on='index')

# create a list that has the new index variable
cluslist=list(full['index'])

# create a list of cluster assignments
labels=list(model2.labels_)

# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))

# convert newlist dictionary to a dataframe
newclus=pd.DataFrame.from_dict(newlist, orient='index')

# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(inplace=True)

merged_train=pd.merge(clus_train, newclus, on='index')

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
merged_train_2=merged_train.copy()
merged_train_2.drop(['index'], axis=1, inplace=True)

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train_2.groupby('cluster').mean()
print (" ")
print (" ")
print ("Clustering variable means by cluster")
print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(clustergrp)
print (" ")
print (" ")

#Plot cluster variable means by cluster
clusterplt= clustergrp[['Region','Fresh', 'Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]
clusterplt.plot.bar()
plt.show()


"""
# validate clusters in training data by examining cluster differences in Region using ANOVA
# first have to merge Region with clustering variables and cluster assignment data 
"""

wretail=wholesale['Channel']
# split Region data into train and test sets
wretail_train, wretail_test = train_test_split(wretail, test_size=.3, random_state=123)
# convert to a dataframe
wretail_train1=pd.DataFrame(wretail)
# intorduce index to uniquely identify data points
wretail_train1.reset_index(inplace=True)
#merge dataframes
merged_train_all=pd.merge(wretail_train1, merged_train, on='index')

sub1 = merged_train_all[['Channel', 'cluster']].dropna()

#function to recode polityscore variable 
def Channel_recode(p):
    if p==1:
        return 0
    else:
        return 1
    
#apply above function to polityscore column
sub1['Channel']=sub1['Channel'].apply(lambda p: Channel_recode(p))
    
# Perform ols regression
import statsmodels.api as sm
import statsmodels.formula.api as smf
reg1=smf.logit(formula='Channel ~ cluster',data=sub1).fit()
#wholesalemod = smf.ols(formula='Channel ~ C(cluster)', data=sub1).fit()
print (reg1.summary())
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
print(' ')
print(' ')

"""
print ('Means for GPA by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for GPA by cluster')
m2= sub1.groupby('cluster').std()
print (m2)
"""

def clc_con_mat_horeca(dic):
    crt_horecas=0
    wrng_horecas=0
    for i in range(1,len(sub1)):
        if dic.iloc[i]['Channel']==0:
            if dic.iloc[i]['Channel']==dic.iloc[i]['cluster']:
                crt_horecas=crt_horecas+1
            else:
                wrng_horecas=wrng_horecas+1
    return(crt_horecas,wrng_horecas)

(crt_horecas,wrng_horecas)=clc_con_mat_horeca(sub1)

def clc_con_mat_retail(dic):
    crt_retail=0
    wrng_retail=0
    for i in range(1,len(sub1)):
        if dic.iloc[i]['Channel']==1:
            if dic.iloc[i]['Channel']==dic.iloc[i]['cluster']:
                crt_retail=crt_retail+1
            else:
                wrng_retail=wrng_retail+1
    return(crt_retail,wrng_retail)

(crt_retail,wrng_retail)=clc_con_mat_horeca(sub1)


print('Number of Horeca Customer Channels in Retail Cluster based on spending patterns:',wrng_horecas)
#print('NUmber of Horeca Customer Channels in Horeca Cluster based on spending patterns',crt_horecas)

print('Number of Retail Customer Channels in Horeca Cluster based on spending patterns:',wrng_retail)
#print('NUmber of Retail Customer Channels in Retail Cluster based on spending patterns',crt_retail)

print(' ')
print(' ')

print ('Standard Deviations for Channel by cluster')
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
m2= sub1.groupby('cluster').std()
print (m2)
print(' ')
print(' ')


print ('Mean for Channel by cluster')
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
m1= sub1.groupby('cluster').std()
print (m1)