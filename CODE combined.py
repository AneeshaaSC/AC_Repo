# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:43:30 2018
@author: 212585611
"""

import os
os.chdir('C:\\Users\\212458792\\Desktop')

import pandas as pd
import numpy as np
from datetime import date
from collections import Counter
from numpy import loadtxt


from sklearn.metrics import r2_score, mean_squared_error
from sklearn import model_selection, preprocessing

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance

import matplotlib.pyplot as plt
#%matplotlib inline

import matplotlib.pylab as plt

##MVP3_testing_Historical_data_June_18_24_25.csvflight_boarding_time
train_df = pd.read_csv('MVP3_testing_Historical_data_June_20_26_27.csv', parse_dates = ['flight_boarding_time'])
train_df['split'] = 1

train_df = train_df.groupby(['flight_number', 'flight_boarding_pt','flight_boarding_time','itemcategory',
           'dishsubcategory','menu_cycle','destination']).first().reset_index()

test_df = pd.read_csv('MVP3_testing_Future_data_June_20_26_27.csv',parse_dates = ['flight_boarding_time'])
test_df['split'] = 0

test_df.Meal = np.nan

#combined_df = train_df
combined_df = train_df.append(test_df, ignore_index=True)

combined_df.sort_values(['flight_number', 'flight_boarding_pt', 'flight_boarding_time'], inplace=True)

lagged_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,30,60]

for i in lagged_list:
    
    combined_df['date_index'] = combined_df.flight_boarding_time
    
    #print(combined_df[['flight_number','flight_boarding_pt','flight_boarding_time','date_index']].head(5))
    
    combined_df.set_index('date_index',inplace=True)
    
    #print(list(combined_df.index))
    #print(combined_df[['flight_number','flight_boarding_pt','date_index']].head(5))
    
    temp1 = combined_df.groupby(['flight_number', 'flight_boarding_pt','itemcategory',
           'dishsubcategory','menu_cycle','destination'])['Meal'].shift(freq='1D',periods=i).reset_index()
    
    combined_df = pd.merge(combined_df.reset_index(),temp1,
                           on = ['flight_number', 'flight_boarding_pt', 'date_index','itemcategory',
                                 'dishsubcategory','menu_cycle','destination'], how = 'left',suffixes=('',i))

zz1 = combined_df[['flight_number', 'flight_boarding_pt','itemcategory','dishsubcategory','menu_cycle','destination']].distinct()
print(zz1.head(5))
    
"""    
    
from pandas.stats.moments import ewma

combined_df['exp_smooth'] = combined_df.groupby(['flight_number', 'flight_boarding_pt',
   'dishsubcategory'])['Meal'].apply(lambda x : ewma(x,span=15))    

combined_df['exp_smooth'] = combined_df.groupby(['flight_number', 'flight_boarding_pt','itemcategory',
       'dishsubcategory'])['exp_smooth'].shift(7)
    
def holt_winters_second_order_ewma( x, span, beta ):
    N = x.size
    alpha = 2.0 / ( 1 + span )
    s = np.zeros(( N, ))
    b = np.zeros(( N, ))
    s[0] = x[0]
    for i in range( 1, N ):
        s[i] = alpha * x[i] + ( 1 - alpha )*( s[i-1] + b[i-1] )
        b[i] = beta * ( s[i] - s[i-1] ) + ( 1 - beta ) * b[i-1]
    return s    
    
combined_df['Doub_exp_smooth'] = combined_df.groupby(['flight_number', 'flight_boarding_pt',
   'dishsubcategory'])['Meal'].apply(lambda x : pd.Series(holt_winters_second_order_ewma(x.values,span=15, beta=0.3))).reset_index()['Meal']

combined_df['Doub_exp_smooth'] = combined_df.groupby(['flight_number', 'flight_boarding_pt','itemcategory',
       'dishsubcategory'])['Doub_exp_smooth'].shift(7)
    
# Live Testing
# Split    
tail = combined_df[combined_df.split==0]

# one hot
cat_vars = ['flight_boarding_pt', 'menu_cycle', 'destination', 
            'itemcategory', 'dishsubcategory']

combined_df = pd.get_dummies(combined_df, columns = cat_vars)
print(combined_df.shape)
train = combined_df[combined_df.split==1]
test = combined_df[combined_df.split==0]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

indep_vars = list(train.columns)
indep_vars = [e for e in indep_vars if e not in ('flight_boarding_time', 'Others', 'Poultry', 
    'Red Meat', 'Seafood', 'Meal',  
     'Meal1', 'Meal2', 'Meal3', 'Meal4', 'Meal5', 'Meal6',
    'dishsubcategory_Poultry','dishsubcategory_Seafood','dishsubcategory_Pasta or Vegetarian',
    'dishsubcategory_Red Meat', 'split', 'date_index')]

target = ['Meal']

X_train = train[indep_vars]
y_train = train[target]

X_test = test[indep_vars]
y_test = test[target]

print(X_train.columns.tolist())

xgb_dump = XGBRegressor(max_depth=8, n_estimators=88, colsample_bytree=0.9, 
                        subsample=0.9, learning_rate=0.05,
                        #reg_alpha=0.02,reg_lambda=2
                        )

xgb_dump.fit(X_train, y_train)

preds = xgb_dump.predict(X_test)

comp = pd.DataFrame()
comp['predictions'] = preds

comp['flight_number'] = tail['flight_number'].tolist()
comp['flight_boarding_pt'] = tail['flight_boarding_pt'].tolist()
comp['flight_boarding_time'] = tail['flight_boarding_time'].tolist()
comp['dishsubcategory'] = tail['dishsubcategory'].tolist()

comp1 = comp[['predictions', 'flight_number', 'flight_boarding_pt',
       'flight_boarding_time']].groupby(['flight_number', 'flight_boarding_pt', 
              'flight_boarding_time']).sum().reset_index()

comp1.columns = ['flight_number', 'flight_boarding_pt', 'flight_boarding_time',
       'Total']

comp = pd.merge(comp,comp1,on=['flight_number', 'flight_boarding_pt',
       'flight_boarding_time'])

comp['percentage_split'] = comp.predictions/comp.Total*100

output = comp[['flight_number', 'flight_boarding_pt','flight_boarding_time', 'dishsubcategory', 'percentage_split']]
#output.to_csv('MVP3_testing_output_sample.csv', index=False)
output.to_csv('MVP3_testing_output_Jun25.csv', index=False)
comp.to_csv('MVP3_testing_output_Jun25_full.csv', index=False)

"""
"""
############################################################
# parameter tuning - retrain once in a while
##########
# test starts from here
pred_date = "2018-05-01 00:00:00"

# valid_date = pred_date - 30 days
valid_date = "2018-04-01 00:00:00"

# valid_date = max date of historical/MOD data + 1
end_date = "2018-06-15 00:00:00"


temp1 = combined_df
#temp1 = temp1[temp1['flight_boarding_time'] <= "2017-12-01 00:00:00"]

train = temp1[temp1['flight_boarding_time'] < valid_date]

validation = temp1[(temp1['flight_boarding_time'] >= valid_date) & (temp1['flight_boarding_time'] < pred_date)]

#tail15 = temp1[temp1['flight_boarding_time'] >= pred_date]
tail15 = temp1[(temp1['flight_boarding_time'] >= pred_date) & 
               (temp1['flight_boarding_time'] < end_date) ]

test = tail15    

# one hot

cat_vars = ['flight_boarding_pt', 'menu_cycle', 'destination', 'itemcategory', 'dishsubcategory']
temp_cols = pd.get_dummies(temp1, columns = cat_vars)

temp_cols_list = list(temp_cols.columns)

train = pd.get_dummies(train, columns = cat_vars).reindex(columns=temp_cols_list,fill_value=0)
validation = pd.get_dummies(validation, columns = cat_vars).reindex(columns=temp_cols_list,fill_value=0)
test = pd.get_dummies(test, columns = cat_vars).reindex(columns=temp_cols_list,fill_value=0)


train.flight_boarding_time.min(),train.flight_boarding_time.max()
validation.flight_boarding_time.min(),validation.flight_boarding_time.max()
test.flight_boarding_time.min(),test.flight_boarding_time.max()


indep_vars = list(train.columns)
indep_vars = [e for e in indep_vars if e not in ('flight_boarding_time', 'Others', 'Poultry', 
    'Red Meat', 'Seafood', 'Meal',  
     'Meal1', 'Meal2', 'Meal3', 'Meal4', 'Meal5', 'Meal6',
    'dishsubcategory_Poultry','dishsubcategory_Seafood','dishsubcategory_Pasta or Vegetarian',
    'dishsubcategory_Red Meat', 'split', 'date_index')]

####

target = ['Meal']

X_train = train[indep_vars]
y_train = train[target]

X_val = validation[indep_vars]
y_val = validation[target]

X_test = test[indep_vars]
y_test = test[target]


xgb_dump = XGBRegressor(max_depth=6, n_estimators=7000, colsample_bytree=0.9, 
                        subsample=0.9, learning_rate=0.05,
                        #reg_alpha=0.02,reg_lambda=2
                        )
xgb_dump.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], 
                                         eval_metric='rmse', verbose=50, early_stopping_rounds=100)

preds = xgb_dump.predict(X_test)
np.sqrt(mean_squared_error(y_test,preds))
    
fig, ax = plt.subplots()
plot_importance(xgb_dump, ax=ax,max_num_features=40)

comp = pd.DataFrame()
comp['predictions'] = preds
comp['actuals'] = y_test['Meal'].tolist()

comp['flight_number'] = tail15['flight_number'].tolist()
comp['flight_boarding_pt'] = tail15['flight_boarding_pt'].tolist()
comp['flight_boarding_time'] = tail15['flight_boarding_time'].tolist()
comp['dishsubcategory'] = tail15['dishsubcategory'].tolist()


comp_P = comp[comp.dishsubcategory == 'Poultry']
comp_S = comp[comp.dishsubcategory == 'Seafood']
comp_R = comp[comp.dishsubcategory == 'Red Meat']
comp_PV = comp[comp.dishsubcategory == 'Pasta or Vegetarian']

np.sqrt(mean_squared_error(comp_P.actuals,comp_P.predictions))
np.sqrt(mean_squared_error(comp_R.actuals,comp_R.predictions))
np.sqrt(mean_squared_error(comp_S.actuals,comp_S.predictions))
np.sqrt(mean_squared_error(comp_PV.actuals,comp_PV.predictions))
          
from sklearn.metrics import mean_absolute_error

#3, 2.6, 3
mean_absolute_error(comp_P.actuals,comp_P.predictions)
mean_absolute_error(comp_R.actuals,comp_R.predictions)
mean_absolute_error(comp_S.actuals,comp_S.predictions)
mean_absolute_error(comp_PV.actuals,comp_PV.predictions)
    
def mae_W(actual, predicted):  
    #actual = np.array(actual.get_label().astype(float))
    mse = np.where(predicted>actual, 0,(actual-predicted))    
    rmse = mse.sum()/len(mse)
    return rmse 

mae_W(comp_P.actuals,comp_P.predictions)
mae_W(comp_R.actuals,comp_R.predictions)
mae_W(comp_S.actuals,comp_S.predictions)
mae_W(comp_PV.actuals,comp_PV.predictions)


comp['values'] = 1

t1 = pd.pivot_table(comp,index = ['flight_number', 'flight_boarding_pt', 'flight_boarding_time', 
                                    ],
               columns = ['dishsubcategory'],
                       values = 'actuals').reset_index()
t1.columns = ['flight_number', 'flight_boarding_pt', 'flight_boarding_time', 'PV_actuals',
       'P_actuals', 'R_actuals', 'S_actuals']

t2 = pd.pivot_table(comp,index = ['flight_number', 'flight_boarding_pt', 'flight_boarding_time', 
                                    ],
               columns = ['dishsubcategory'],
                       values = 'predictions',  
                       aggfunc=np.sum).reset_index()
t2.columns = ['flight_number', 'flight_boarding_pt', 'flight_boarding_time', 'PV_predictions',
       'P_predictions', 'R_predictions', 'S_predictions']

comp = pd.concat([t1, t2.drop(['flight_number', 'flight_boarding_pt', 'flight_boarding_time', 
                                    ],axis=1)],axis=1)
    
#comp1.to_csv('LAGGED_RESULTS_expo_only.CSV', index=False)

comp.fillna(0,inplace=True)
#3.9, 3.3, 4.2
np.sqrt(mean_squared_error(comp.P_actuals,comp.P_predictions))
np.sqrt(mean_squared_error(comp.R_actuals,comp.R_predictions))
np.sqrt(mean_squared_error(comp.S_actuals,comp.S_predictions))
np.sqrt(mean_squared_error(comp.PV_actuals,comp.PV_predictions))

from sklearn.metrics import mean_absolute_error

#3, 2.6, 3
mean_absolute_error(comp.P_actuals,comp.P_predictions)
mean_absolute_error(comp.R_actuals,comp.R_predictions)
mean_absolute_error(comp.S_actuals,comp.S_predictions)    
mean_absolute_error(comp.PV_actuals,comp.PV_predictions)        

 
zz = combined_df[combined_df.Meal.isnull()]

#FRA	Poultry	DXB	2018-06-08 08:25:00	45.0
zz1 = combined_df[(combined_df.flight_number == 45) & (combined_df.flight_boarding_time == '2018-06-08 08:25:00')]
"""