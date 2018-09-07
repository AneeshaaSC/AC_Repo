"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x:'%f'%x)

import matplotlib.pyplot as plt
import seaborn as sb



yl=[397,470,446,392,467,434]
xl=[' ','Right Upper','Right Middle','Right Lower','Left Upper','Left Middle','Left Lower']


fig = plt.figure()
#fig.subtitle('Algorithm Comparison')
ax = fig.add_subplot(111)
y_pos = np.arange(len(xl)-1)
ax.set_xticklabels(xl)
sca=ax
plt.bar(y_pos,yl,color='b')
#plt.plot(yl,linestyle='--', marker='o', color='b')
#plt.xticks(yl,xl,rotation=40)
plt.xlabel('Zone Name')
plt.ylabel('Patient Count')
plt.title('Count of Observations in different Zones')
plt.show()



yl=[25,34,36,32,50,84]
xl=[' ','Right Upper','Right Middle','Right Lower','Left Upper','Left Middle','Left Lower']


fig = plt.figure()
#fig.subtitle('Algorithm Comparison')
ax = fig.add_subplot(111)
y_pos = np.arange(len(xl)-1)
ax.set_xticklabels(xl)
sca=ax
plt.bar(y_pos,yl,color='b')
#plt.plot(yl,linestyle='--', marker='o', color='b')
#plt.xticks(yl,xl,rotation=40)
plt.xlabel('Zone Name')
plt.ylabel('Number of Outliers')
plt.title('Count of Outliers in different Zones')
plt.show()

"""

import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 6
train_acc = (90.61, 89.66, 87.82, 92.7,82.82,86.13)
test_acc = (88.33, 88.65, 86.56,91.52,83.68,85.49 )
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, train_acc, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Training Set')
 
rects2 = plt.bar(index + bar_width, test_acc, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test Set')
 
plt.xlabel('Zone Names')
plt.ylabel('Accuracy %')
plt.title('Model Accuracy across Zones')
plt.xticks(index + bar_width, ('Zone-1', 'Zone-2', 'Zone-3', 'Zone-4','Zone-5','Zone-6'))
plt.legend()
 
plt.tight_layout()
plt.show()