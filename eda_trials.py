import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sb

#Format the display a little bit
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x:'%f'%x)

# read data from all different tabs in excel 
comb=pd.read_excel('combined.xlsx')
#plt.figure(figsize=(12, 9))
comb['Label'] = comb['Label'].convert_objects(convert_numeric=True)

numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')

numoru['Hist_0_0_0_Entropy_Bins']=pd.qcut(numoru.Hist_0_0_0_Entropy,4,labels=["Very Low","Low","Average","High"])
numorm['Hist_0_0_0_Entropy_Bins']=pd.qcut(numorm.Hist_0_0_0_Entropy,4,labels=["Very Low","Low","Average","High"])
numorl['Hist_0_0_0_Entropy_Bins']=pd.qcut(numorl.Hist_0_0_0_Entropy,4,labels=["Very Low","Low","Average","High"])
numolu['Hist_0_0_0_Entropy_Bins']=pd.qcut(numolu.Hist_0_0_0_Entropy,4,labels=["Very Low","Low","Average","High"])
numolm['Hist_0_0_0_Entropy_Bins']=pd.qcut(numolm.Hist_0_0_0_Entropy,4,labels=["Very Low","Low","Average","High"])
numoll['Hist_0_0_0_Entropy_Bins']=pd.qcut(numoll.Hist_0_0_0_Entropy,4,labels=["Very Low","Low","Average","High"])

#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(comb, alpha=0.2, figsize=(60, 60), diagonal='kde')
# pairwise correlation
#print(comb.drop(['PatientNumMasked', 'Label'], axis=1).corr(method='spearman'))
"""
fig = plt.figure(figsize=(10, 20))
ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6 = fig.add_subplot(336)
ax7 = fig.add_subplot(337)
ax8 = fig.add_subplot(338)
ax9 = fig.add_subplot(339)

plt.subplots_adjust(hspace = 1, wspace=0.4)

sb.factorplot(x="Hist_2_150_2_Entropy_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax1)
sb.factorplot(x="Hist_2_150_2_Skewness_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax2)
sb.factorplot(x="CoMatrix_Deg90_Local_Homogeneity_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax3)


sb.factorplot(x="CoMatrix_Deg135_Local_Homogeneity_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax4)
sb.factorplot(x="CoMatrix_Deg135_Correlation_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax5)
sb.factorplot(x="CoMatrix_Deg135_Inertia_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax6)

sb.factorplot(x="Hist_0_0_0_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax7)
sb.factorplot(x="Hist_0_0_0_Skewness_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax8)
sb.factorplot(x="Hist_2_45_1_Entropy_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax9)

comb['Hist_2_180_2_Kurtosis_Bins']=pd.qcut(comb.Hist_2_180_2_Kurtosis,4,labels=["Very Low","Low","Average","High"])
comb['Hist_2_180_2_Mean_Bins']=pd.qcut(comb.Hist_2_180_2_Mean,4,labels=["Very Low","Low","Average","High"])
comb['Hist_2_60_1_Skewness_Bins']=pd.qcut(comb.Hist_2_60_1_Skewness,4,labels=["Very Low","Low","Average","High"])

comb['Hist_1_180_2_Mean_Bins']=pd.qcut(comb.Hist_1_180_2_Mean,4,labels=["Very Low","Low","Average","High"])
comb['Hist_1_180_2_StdDev_Bins']=pd.qcut(comb.Hist_1_180_2_StdDev,4,labels=["Very Low","Low","Average","High"])
comb['Hist_1_120_2_Mean_Bins']=pd.qcut(comb.Hist_1_120_2_Mean ,4,labels=["Very Low","Low","Average","High"])

comb['Hist_1_30_2_Mean_Bins']=pd.qcut(comb.Hist_1_30_2_Mean ,4,labels=["Very Low","Low","Average","High"])
comb['Hist_2_30_2_Mean_Bins']=pd.qcut(comb.Hist_2_30_2_Mean ,4,labels=["Very Low","Low","Average","High"])                                                                                                                 
comb['Hist_1_135_2_Mean_Bins']=pd.qcut(comb.Hist_1_135_2_Mean ,4,labels=["Very Low","Low","Average","High"])


fig2 = plt.figure(figsize=(10, 20))
plt.subplots_adjust(hspace = 0.8, wspace=0.4)
ax21 = fig2.add_subplot(531)
ax22 = fig2.add_subplot(532)
ax23 = fig2.add_subplot(533)
ax24 = fig2.add_subplot(534)
ax25 = fig2.add_subplot(535)
ax26 = fig2.add_subplot(536)
ax27 = fig2.add_subplot(537)
ax28 = fig2.add_subplot(538)
ax29 = fig2.add_subplot(539)

sb.factorplot(x="Hist_2_180_2_Kurtosis_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax21)
sb.factorplot(x="Hist_2_180_2_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax22)
sb.factorplot(x="Hist_2_60_1_Skewness_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax23)
sb.factorplot(x="Hist_1_180_2_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax24)
sb.factorplot(x="Hist_1_180_2_StdDev_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax25)
sb.factorplot(x="Hist_1_120_2_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax26)
sb.factorplot(x="Hist_1_30_2_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax27)
sb.factorplot(x="Hist_2_30_2_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax28)
sb.factorplot(x="Hist_1_135_2_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax29)


comb['Hist_2_90_1_Skewness_Bins']=pd.qcut(comb.Hist_2_90_1_Skewness ,4,labels=["Very Low","Low","Average","High"])
comb['Hist_1_90_2_Skewness_Bins']=pd.qcut(comb.Hist_1_90_2_Skewness ,4,labels=["Very Low","Low","Average","High"])                                                                                                                 
comb['Hist_2_90_2_Mean_Bins']=pd.qcut(comb.Hist_2_90_2_Mean ,4,labels=["Very Low","Low","Average","High"])



fig3  = plt.figure(figsize=(12,3))
plt.subplots_adjust(wspace=0.4)
ax31 = fig3.add_subplot(131)
ax32 = fig3.add_subplot(132)
ax33 = fig3.add_subplot(133)

sb.factorplot(x="Hist_2_90_1_Skewness_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax31)
sb.factorplot(x="Hist_1_90_2_Skewness_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax32)
sb.factorplot(x="Hist_2_90_2_Mean_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax33)

comb['Hist_1_135_2_Entropy_Bins']=pd.qcut(comb.Hist_1_135_2_Entropy ,4,labels=["Very Low","Low","Average","High"])

fig4  = plt.figure(figsize=(3,3))
#plt.subplots_adjust()
ax41 = fig4.add_subplot(111)
sb.factorplot(x="Hist_1_135_2_Entropy_Bins", y="Label", data=comb, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax41)
"""

#import pandas_profiling 

#pandas_profiling.ProfileReport(comb)
comb.drop(['PatientNumMasked', 'Label'], axis=1)
numoru.drop(['PatientNumMasked', 'LabelRU'], axis=1)
"""
c=comb.corr()
sb.heatmap(c, 
        xticklabels=c.columns,
        yticklabels=c.columns)
 
"""
"""
def correlation_matrix(df):

    from matplotlib import cm as cm

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    labels=['Hist_0_0_0_Mean','Hist_0_0_0_Skewness','Hist_0_0_0_Kurtosis','Hist_0_0_0_Entropy','Hist_2_45_1_Entropy','Hist_2_60_1_Skewness','Hist_2_90_1_Skewness','Hist_2_90_1_Kurtosis','Hist_2_135_1_Entropy','Hist_1_150_1_Skewness','Hist_2_180_1_Skewness','Hist_1_30_2_Mean','Hist_2_30_2_Mean','Hist_2_30_2_Entropy','Hist_2_60_2_Skewness','Hist_2_60_2_Kurtosis','Hist_1_90_2_Skewness','Hist_2_90_2_Mean','Hist_2_90_2_Skewness','Hist_2_90_2_Kurtosis','Hist_1_120_2_Mean','Hist_1_135_2_Mean','Hist_1_135_2_Entropy','Hist_2_150_2_Mean','Hist_2_150_2_Skewness','Hist_2_150_2_Kurtosis','Hist_2_150_2_Entropy','Hist_1_180_2_Mean','Hist_1_180_2_StdDev','Hist_1_180_2_Skewness','Hist_2_180_2_Mean','Hist_2_180_2_Skewness','Hist_2_180_2_Kurtosis','Hist_2_180_2_Entropy','CoMatrix_Deg45_Local_Homogeneity','CoMatrix_Deg90_Local_Homogeneity','CoMatrix_Deg135_Local_Homogeneity','CoMatrix_Deg135_Correlation','CoMatrix_Deg135_Inertia']
    ax1.set_xticklabels(labels,fontsize=9)
    plt.xticks(rotation=90)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show(block=True)

correlation_matrix(numoru)

"""
f, ax = plt.subplots(figsize=(12, 9))
corr = numoru.corr()

sb.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sb.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
"""
plt.matshow(numoru.corr())

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
plot_corr(comb)
"""