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

import matplotlib.pyplot as plt
import seaborn as sb



# read data from all different tabs in excel 
numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')

numoru['Hist_2_60_1_Skewness_Bins']=pd.qcut(numoru.Hist_2_60_1_Skewness,4,labels=["Very Low","Low","Average","High"])
numorm['Hist_2_60_1_Skewness_Bins']=pd.qcut(numorm.Hist_2_60_1_Skewness,4,labels=["Very Low","Low","Average","High"])
numorl['Hist_2_60_1_Skewness_Bins']=pd.qcut(numorl.Hist_2_60_1_Skewness,4,labels=["Very Low","Low","Average","High"])
numolu['Hist_2_60_1_Skewness_Bins']=pd.qcut(numolu.Hist_2_60_1_Skewness,4,labels=["Very Low","Low","Average","High"])
numolm['Hist_2_60_1_Skewness_Bins']=pd.qcut(numolm.Hist_2_60_1_Skewness,4,labels=["Very Low","Low","Average","High"])
numoll['Hist_2_60_1_Skewness_Bins']=pd.qcut(numoll.Hist_2_60_1_Skewness,4,labels=["Very Low","Low","Average","High"])

fig = plt.figure(figsize=(10, 20))
ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6 = fig.add_subplot(336)
plt.subplots_adjust(hspace=0.8,wspace=0.8)

sb.factorplot(x="Hist_2_60_1_Skewness_Bins", y="LabelRU", data=numoru, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax1)
sb.factorplot(x="Hist_2_60_1_Skewness_Bins", y="LabelRM", data=numorm, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax2)
sb.factorplot(x="Hist_2_60_1_Skewness_Bins", y="LabelRL", data=numorl, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax3)


sb.factorplot(x="Hist_2_60_1_Skewness_Bins", y="LabelLU", data=numolu, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax4)
sb.factorplot(x="Hist_2_60_1_Skewness_Bins", y="LabelLM", data=numolm, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax5)
sb.factorplot(x="Hist_2_60_1_Skewness_Bins", y="LabelLL", data=numoll, kind="bar", ci=None,x_order=['Very Low','Low','Average','High'],ax=ax6)
