# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:20:45 2017

@author: 212458792
"""

l=[88.377,90.35,91.28,90.67]
n=[' ','Logistic Regression',' ','K-Nearest Neighbours',' ','Extra Trees Classifier',' ','Support Vector Machines']

import matplotlib.pyplot as plt
import seaborn as sb

#sb.regplot(n,l, scatter_kws={"s": 100})

fig = plt.figure()
#fig.subtitle('Algorithm Comparison')
ax = fig.add_subplot(111)
ax.set_xticklabels(n)
sca=ax
plt.plot(l,linestyle='--', marker='o', color='b')
plt.xticks(rotation=40)
#print(len(l))
plt.title('Model Comparison')
plt.show()


