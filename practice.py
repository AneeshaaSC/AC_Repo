# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:08:02 2017

@author: 212458792
"""

import pandas as pd
import numpy as np

"""
pred_train, pred_test, tar_train, tar_test,tarpat,testpat  = train_test_split(dfpred, dftar,dfpat, test_size=.3,random_state=123)


sales = [{'account': 'Jones LLC', 'Jan': 150, 'Feb': 200, 'Mar': 140,'label': 1,'index':1},
         {'account': 'Alpha Co',  'Jan': 200, 'Feb': 210, 'Mar': 215,'label':0,'index':2},
         {'account': 'Blue Inc',  'Jan': 50,  'Feb': 90,  'Mar': 95,'label':0,'index':3 }]
df = pd.DataFrame(sales)
#print(df)


from sklearn.model_selection import LeaveOneOut 
#X = np.array([[1, 2], [3, 4]])
y = np.array([1,2,3,4])
loo = LeaveOneOut()
#print(loo.get_n_splits(sales))

predic=[]
for train_index, test_index in loo.split(df):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = df.iloc[train_index], df.iloc[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train)
   print('---')
   print(X_test)
   print(X_test['index'])
   
   
#print(predic)
  
z = np.array([[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]])

pd.DataFrame(data=z[1:,1:],        
               index=z[:,0])
#df = pd.DataFrame(z)
print(z)

   
print(z.shape)

l=z.reshape(-1,1)
print(l)   
print(l.shape)
print(z[0:5,0])

s=[1,0]
print(s)

s=[[1],[0]]
print(s)


A = np.random.randn(4,3)
print(A)
B = np.sum(A, axis = 1, keepdims = True)
print(B.shape)
print(B)

a=[[1, 2, 3],[1, 2, 3]]

b=[[4,5,6],[4,5,6]]

print(np.multiply(a,b))

print(type(a))
print(type(b))

for i in range(4):
    print(i)

"""

def solution(A):
    # write your code in Python 3.6
    s=1
    tocontinue = True
    while tocontinue:
        if s in A:
            s=s+1
        else:
            tocontinue=False
    return s
    
print(solution([-1,-3]))
