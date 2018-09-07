# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:38:33 2017

@author: 212458792
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

#Format the display a little bit
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x:'%f'%x)

# read data from all different tabs in excel 
numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')

#print(type(numoru))


masterid=numoru[['PatientNumMasked','LabelRU']]
masterid=pd.merge(masterid, numorm[['PatientNumMasked','LabelRM']], on='PatientNumMasked')
masterid=pd.merge(masterid, numorl[['PatientNumMasked','LabelRL']], on='PatientNumMasked')
masterid=pd.merge(masterid, numolu[['PatientNumMasked','LabelLU']], on='PatientNumMasked')
masterid=pd.merge(masterid, numolm[['PatientNumMasked','LabelLM']], on='PatientNumMasked')
masterid=pd.merge(masterid, numoll[['PatientNumMasked','LabelLL']], on='PatientNumMasked')

print('data from all zones merged  ... ')
print('No. of patients with data for all six zone',masterid.shape)
masterid['LabelRM'] =pd.to_numeric(masterid['LabelRM'], errors='coerce')
masterid['LabelRL'] =pd.to_numeric(masterid['LabelRL'], errors='coerce')
masterid['LabelLU'] =pd.to_numeric(masterid['LabelLU'], errors='coerce')
masterid['LabelRU'] =pd.to_numeric(masterid['LabelRU'], errors='coerce')
masterid['LabelLM'] =pd.to_numeric(masterid['LabelLM'], errors='coerce')
masterid['LabelLL'] =pd.to_numeric(masterid['LabelLL'], errors='coerce')
masterid['LabelLL'] =pd.to_numeric(masterid['LabelLL'], errors='coerce')
#print(masterid)

def chk_uniformity():
    l=[]
    for i in range(len(masterid)):
        s=masterid.iloc[i]['LabelRU']+masterid.iloc[i]['LabelRM']+masterid.iloc[i]['LabelRL']+masterid.iloc[i]['LabelLU']+masterid.iloc[i]['LabelLM']+masterid.iloc[i]['LabelLL']
        #print(s)
        #print(masterid.iloc[i]['Patientid'])
        if s>0 and s<6:
            l.append(masterid.iloc[i]['PatientNumMasked'])
    if len(l)==0:
        print('All patients have same labels in all zones')
    else:
        return l

print('calling function ..')

print(chk_uniformity())
"""
print('Number of Abnormal and Normal Patients across six Zones')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(' ')
print('Zone-1')
print(numoru['LabelRU'].value_counts())
print(' ')
print('Zone-2')
print(numorm['LabelRM'].value_counts())
print(' ')
print('Zone-3')
print(numorl['LabelRL'].value_counts())
print(' ')
print('Zone-4')
print(numolu['LabelLU'].value_counts())
print(' ')
print('Zone-5')
print(numolm['LabelLM'].value_counts())
print(' ')
print('Zone-6')
print(numoll['LabelLL'].value_counts())




plt.figure(figsize=(12, 9))
plt.subplot(3, 2, 1)
numoru['LabelRU'].value_counts().plot(kind='bar')
plt.title('Zone-1')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
 
plt.subplot(3, 2, 2)
numorm['LabelRM'].value_counts().plot(kind='bar')
plt.title('Zone-2')


plt.xticks(rotation=0)
 
plt.subplot(3, 2, 3)
numorl['LabelRL'].value_counts().plot(kind='bar')
plt.title('Zone-3')

plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
 
plt.subplot(3, 2, 4)
numolu['LabelLU'].value_counts().plot(kind='bar')
plt.title('Zone-4')


plt.xticks(rotation=0)
 

plt.subplot(3, 2, 5)
numolm['LabelLM'].value_counts().plot(kind='bar')
plt.title('Zone-5')
plt.xlabel('Patient Label')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
 
plt.subplot(3, 2, 6)
numoll['LabelLL'].value_counts().plot(kind='bar')
plt.title('Zone-6')
plt.xlabel('Patient Label')

plt.xticks(rotation=0)

#plt.subplots_adjust(wspace=0.25, hspace=0.5)
plt.show()

        
"""
 

#All patients have same label in all zones


import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing

numolu_cp = numolu+0.00001*np.random.rand(numolu.shape[0], numolu.shape[1])
numolu_cp['LabelLU']=numolu['LabelLU']
numolu_cp['PatientNumMasked']=numolu['PatientNumMasked']


#numolu_cp['Hist_0_0_0_Mean']=preprocessing.scale(numolu_cp['Hist_0_0_0_Mean'].astype('float64'))
numolm['Hist_0_0_0_Skewness']=preprocessing.scale(numolm['Hist_0_0_0_Skewness'].astype('float64'))
numoll['Hist_0_0_0_Skewness']=preprocessing.scale(numoll['Hist_0_0_0_Skewness'].astype('float64'))
numolu['Hist_0_0_0_Skewness']=preprocessing.scale(numolu['Hist_0_0_0_Skewness'].astype('float64'))

numorm['Hist_0_0_0_Skewness']=preprocessing.scale(numorm['Hist_0_0_0_Skewness'].astype('float64'))
numorl['Hist_0_0_0_Skewness']=preprocessing.scale(numorl['Hist_0_0_0_Skewness'].astype('float64'))
numoru['Hist_0_0_0_Skewness']=preprocessing.scale(numoru['Hist_0_0_0_Skewness'].astype('float64'))


"""
numolu_cp['Hist_0_0_0_Kurtosis']=preprocessing.scale(numolu_cp['Hist_0_0_0_Kurtosis'].astype('float64'))
numolu_cp['Hist_0_0_0_Entropy']=preprocessing.scale(numolu_cp['Hist_0_0_0_Entropy'].astype('float64'))
numolu_cp['Hist_2_45_1_Entropy']=preprocessing.scale(numolu_cp['Hist_2_45_1_Entropy'].astype('float64'))
numolu_cp['Hist_2_60_1_Skewness']=preprocessing.scale(numolu_cp['Hist_2_60_1_Skewness'].astype('float64'))
numolu_cp['Hist_2_90_1_Skewness']=preprocessing.scale(numolu_cp['Hist_2_90_1_Skewness'].astype('float64'))
numolu_cp['Hist_2_90_1_Kurtosis']=preprocessing.scale(numolu_cp['Hist_2_90_1_Kurtosis'].astype('float64'))
numolu_cp['Hist_2_135_1_Entropy']=preprocessing.scale(numolu_cp['Hist_2_135_1_Entropy'].astype('float64'))
numolu_cp['Hist_1_150_1_Skewness']=preprocessing.scale(numolu_cp['Hist_1_150_1_Skewness'].astype('float64'))
numolu_cp['Hist_2_180_1_Skewness']=preprocessing.scale(numolu_cp['Hist_2_180_1_Skewness'].astype('float64'))
numolu_cp['Hist_1_30_2_Mean']=preprocessing.scale(numolu_cp['Hist_1_30_2_Mean'].astype('float64'))
numolu_cp['Hist_2_30_2_Mean']=preprocessing.scale(numolu_cp['Hist_2_30_2_Mean'].astype('float64'))
numolu_cp['Hist_2_30_2_Entropy']=preprocessing.scale(numolu_cp['Hist_2_30_2_Entropy'].astype('float64'))
numolu_cp['Hist_2_60_2_Skewness']=preprocessing.scale(numolu_cp['Hist_2_60_2_Skewness'].astype('float64'))
numolu_cp['Hist_2_60_2_Kurtosis']=preprocessing.scale(numolu_cp['Hist_2_60_2_Kurtosis'].astype('float64'))
numolu_cp['Hist_1_90_2_Skewness']=preprocessing.scale(numolu_cp['Hist_1_90_2_Skewness'].astype('float64'))
numolu_cp['Hist_2_90_2_Mean']=preprocessing.scale(numolu_cp['Hist_2_90_2_Mean'].astype('float64'))
numolu_cp['Hist_2_90_2_Skewness']=preprocessing.scale(numolu_cp['Hist_2_90_2_Skewness'].astype('float64'))
numolu_cp['Hist_2_90_2_Kurtosis']=preprocessing.scale(numolu_cp['Hist_2_90_2_Kurtosis'].astype('float64'))
numolu_cp['Hist_1_120_2_Mean']=preprocessing.scale(numolu_cp['Hist_1_120_2_Mean'].astype('float64'))
numolu_cp['Hist_1_135_2_Mean']=preprocessing.scale(numolu_cp['Hist_1_135_2_Mean'].astype('float64'))
numolu_cp['Hist_1_135_2_Entropy']=preprocessing.scale(numolu_cp['Hist_1_135_2_Entropy'].astype('float64'))
numolu_cp['Hist_2_150_2_Mean']=preprocessing.scale(numolu_cp['Hist_2_150_2_Mean'].astype('float64'))
numolu_cp['Hist_2_150_2_Skewness']=preprocessing.scale(numolu_cp['Hist_2_150_2_Skewness'].astype('float64'))
numolu_cp['Hist_2_150_2_Kurtosis']=preprocessing.scale(numolu_cp['Hist_2_150_2_Kurtosis'].astype('float64'))
numolu_cp['Hist_2_150_2_Entropy']=preprocessing.scale(numolu_cp['Hist_2_150_2_Entropy'].astype('float64'))
numolu_cp['Hist_1_180_2_Mean']=preprocessing.scale(numolu_cp['Hist_1_180_2_Mean'].astype('float64'))
numolu_cp['Hist_1_180_2_StdDev']=preprocessing.scale(numolu_cp['Hist_1_180_2_StdDev'].astype('float64'))
numolu_cp['Hist_1_180_2_Skewness']=preprocessing.scale(numolu_cp['Hist_1_180_2_Skewness'].astype('float64'))
numolu_cp['Hist_2_180_2_Mean']=preprocessing.scale(numolu_cp['Hist_2_180_2_Mean'].astype('float64'))
numolu_cp['Hist_2_180_2_Skewness']=preprocessing.scale(numolu_cp['Hist_2_180_2_Skewness'].astype('float64'))
numolu_cp['Hist_2_180_2_Kurtosis']=preprocessing.scale(numolu_cp['Hist_2_180_2_Kurtosis'].astype('float64'))
numolu_cp['Hist_2_180_2_Entropy']=preprocessing.scale(numolu_cp['Hist_2_180_2_Entropy'].astype('float64'))
numolu_cp['CoMatrix_Deg45_Local_Homogeneity']=preprocessing.scale(numolu_cp['CoMatrix_Deg45_Local_Homogeneity'].astype('float64'))
numolu_cp['CoMatrix_Deg90_Local_Homogeneity']=preprocessing.scale(numolu_cp['CoMatrix_Deg90_Local_Homogeneity'].astype('float64'))
numolu_cp['CoMatrix_Deg135_Local_Homogeneity']=preprocessing.scale(numolu_cp['CoMatrix_Deg135_Local_Homogeneity'].astype('float64'))
numolu_cp['CoMatrix_Deg135_Correlation']=preprocessing.scale(numolu_cp['CoMatrix_Deg135_Correlation'].astype('float64'))
numolu_cp['CoMatrix_Deg135_Inertia']=preprocessing.scale(numolu_cp['CoMatrix_Deg135_Inertia'].astype('float64'))
"""

#print(numoru['LabelRU'].value_counts())
reg1=smf.logit(formula='LabelLU ~ Hist_0_0_0_Skewness',data=numolu).fit()
print(reg1.summary())
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

reg2=smf.logit(formula='LabelLM ~ Hist_0_0_0_Skewness',data=numolm).fit()
print(reg2.summary())
print(' ')
print(' ')
print('Odds Ratio')
print('-----------')
#print(np.exp(reg1.params)) 


params = reg2.params
conf = reg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

reg3=smf.logit(formula='LabelLL ~ Hist_0_0_0_Skewness+Hist_0_0_0_Entropy',data=numoll).fit()
print(reg3.summary())
print(' ')
print(' ')
print('Odds Ratio')
print('-----------')
#print(np.exp(reg1.params)) 


params = reg3.params
conf = reg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))


"""
# Create correlation matrix
corr_matrix = numolu.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
print(upper)
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print('correlated columns')
print(to_drop)
"""
