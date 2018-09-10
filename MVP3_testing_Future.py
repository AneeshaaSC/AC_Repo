# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:17:31 2018

@author: 212585611
"""

import os
os.chdir('C:\\Users\\212585611\\Downloads\\Aviation\\v2')

import pandas as pd
import numpy as np
from datetime import date
from collections import Counter
from numpy import loadtxt

# new file - spark_pax_flight_future_final
#spark_pax_flight_future_final_19thJune
spark_future_pax_data = pd.read_csv('spark_pax_flight_future_final_21_27_28_June_1.csv', parse_dates = ['spark_pax_flight_future_final.flight_boarding_time'])
spark_future_pax_data.columns = [w.replace('spark_pax_flight_future_final.','') for w in spark_future_pax_data.columns]

#spark_future_pax_data = spark_future_pax_data[spark_future_pax_data.flight_boarding_time >= '2018-06-15 00:00:00']

# old file
#spark_future_pax_data = pd.read_csv('spark_future_pax_data.csv', parse_dates = ['spark_future_pax_data.flight_boarding_time'])
#spark_future_pax_data.columns = [w.replace('spark_future_pax_data.','') for w in spark_future_pax_data.columns]

spark_future_pax_data.rename(columns = {'board_point':'flight_boarding_pt', 
                                        'menu_name': 'menuname'}, inplace = True)

# temp check to see if there any new dishes with disubcategory as null
# add the category manually if there are any
#M1 = spark_future_pax_data[(spark_future_pax_data.meal_service_name == 'Hot Meal') & (spark_future_pax_data.dishcategory == 'Main Course')]
#Counter(M1.dishsubcategory)


spark_future_pax_data.dishsubcategory[spark_future_pax_data.menucardname == 'Cajun chicken'] = 'Poultry'

#M1 = spark_future_pax_data[((spark_future_pax_data.meal_service_name == 'Hot Meal') | ((spark_future_pax_data.flight_number == 306) & (spark_future_pax_data.meal_service_name == 'Hot Breakfast'))) & (spark_future_pax_data.dishcategory == 'Main Course')]
M1 = spark_future_pax_data[(spark_future_pax_data.meal_service_name == 'Hot Meal') & (spark_future_pax_data.dishcategory == 'Main Course')]

#M1 = spark_future_pax_data[(spark_future_pax_data.meal_service_name.isin('Hot Meal', '') & (spark_future_pax_data.dishcategory == 'Main Course')]

# Cuisine
cuisine = pd.read_csv('Dish_Cuisine_Sandra.csv', encoding='latin').drop_duplicates()
cuisine.Cuisine = cuisine.Cuisine.replace('\?','',regex=True)
cuisine.itemname = cuisine.itemname.str.lower()
cuisine.itemname = cuisine.itemname.str.strip()

cuisine.Cuisine = cuisine.Cuisine.str.lower()
cuisine.Cuisine = cuisine.Cuisine.str.strip()

M1.menucardname = M1.menucardname.str.lower()
M1.menucardname = M1.menucardname.str.strip()

M1 = pd.merge(M1,cuisine,left_on = 'menucardname', right_on = 'itemname', how = 'left').shape

# Menu cycle, Destination
M1['menuname'][M1['menuname'] == 'F DXBAUS HM J Q A'] = 'DXBAUS HM J Q A'
M1['menuname'][M1['menuname'] == 'FEST2017 DXBEUR HMJ'] = 'DXBEUR HMJ FEST2017'
M1['menuname'][M1['menuname'] == 'FEST2017 DXBGER HMJ'] = 'DXBGER HMJ FEST2017'
M1['menuname'][M1['menuname'] == 'HO 2017 DXBCDG HM JB'] = 'DXBCDG HM JB'
M1['menuname'][M1['menuname'] == 'TR DXBMEL HM J T2'] = 'DXBMEL HM J T'

M1['menu_cycle'] = M1['menuname'].str.split().str[-1]
M1['menu_cycle'][M1['menu_cycle'] == 'JA'] = 'A'
M1['menu_cycle'][M1['menu_cycle'] == 'JB'] = 'B'
M1['menu_cycle'][M1['menu_cycle'] == 'FEST17'] = 'FEST2017'

M1['destination'] = M1['menuname'].str.split().str[0].str[3:]

# Age group
M1['date_of_birth'] = pd.to_datetime(M1['date_of_birth'])

M1['today_date'] = date.today()
M1['today_date'] = pd.to_datetime(M1['today_date'])

M1['age'] = (M1['today_date']-M1['date_of_birth'])/np.timedelta64(1, 'Y')

bins = [0, 12, 19, 40, 60, 100]
M1['age_groups'] = pd.cut(M1['age'], bins)

M1['age_group'] = M1['age_groups'].cat.codes

M1['age_group_1'] = np.nan 
M1['age_group_1'][M1['age_group'] == 4] = 'Elders' 
M1['age_group_1'][M1['age_group'] == 3] = 'Middle Aged' 
M1['age_group_1'][M1['age_group'] == 2] = 'Adults' 
M1['age_group_1'][M1['age_group'] == 1] = 'Teenagers'
M1['age_group_1'][M1['age_group'] == 0] = 'Children' 

# Country
country_regions = pd.read_csv('country_codes_cuisine.csv',keep_default_na = False,na_values=[''])
M1 = pd.merge(M1,country_regions, left_on = 'nationality', right_on = 'alpha-2', how = 'left')

M1['values'] = 1

# dishsubcategory - flight level aggregations
temp1_dishsub_cuisine = pd.get_dummies(M1[['flight_number', 'flight_boarding_pt', 
                           'flight_boarding_time', 'dishsubcategory','Cuisine']].drop_duplicates(),
columns = ['dishsubcategory','Cuisine'])

#.groupby(['flight_number', 'flight_boarding_pt', 
#                           'flight_boarding_time']).sum().reset_index()


# demographics - flight level aggregations

pl = ['flight_number', 'flight_boarding_pt', 'flight_boarding_time','destination', 
      'menu_cycle', 'service_category_code']

columns_list = ['gender', 'country_region']

temp1_demographics = pd.pivot_table(M1[['flight_number', 'flight_boarding_pt', 
                           'flight_boarding_time','pax_id','menu_cycle','destination', 'service_category_code',
                           'age_group_1','gender', 'country_region','values']].drop_duplicates(), 
index = pl, columns = 'age_group_1',
                       values = 'values',  
                       aggfunc=np.sum).reset_index()

for i in columns_list:
    temp1_demo = pd.pivot_table(M1[['flight_number', 'flight_boarding_pt', 
                               'flight_boarding_time','pax_id','menu_cycle','destination', 'service_category_code',
                               'age_group_1','gender', 'country_region','values']].drop_duplicates(), 
    index = pl, columns = i,
                           values = 'values',  
                           aggfunc=np.sum).reset_index()
    temp1_demographics = pd.concat([temp1_demographics, temp1_demo.drop(pl,axis=1)], axis=1)
    
temp1 = pd.merge(temp1_demographics,temp1_dishsub_cuisine, on = ['flight_number', 'flight_boarding_pt', 
                               'flight_boarding_time'])

# extract basic features from date
temp1['year'] = temp1['flight_boarding_time'].dt.year
temp1['month'] = temp1['flight_boarding_time'].dt.month
temp1['quarter'] = temp1['flight_boarding_time'].dt.quarter
temp1['week'] = temp1['flight_boarding_time'].dt.week
temp1['day'] = temp1['flight_boarding_time'].dt.day
temp1['dayofweek'] = temp1['flight_boarding_time'].dt.dayofweek

temp1.columns = [w.replace('dishsubcategory_','') for w in temp1.columns]
temp1.columns = [w.replace('Cuisine_','') for w in temp1.columns]

# wide to long
temp1.rename(columns = {'Poultry':'Meal_Poultry',
       'Red Meat':'Meal_Red Meat', 'Seafood':'Meal_Seafood', 'Pasta or Vegetarian': 'Meal_Pasta or Vegetarian'}, inplace = True)

list_melt = temp1.columns.tolist()
list_melt = [e for e in list_melt if e not in ('Meal_Others',
       'Meal_Poultry', 'Meal_Red Meat', 'Meal_Seafood','Meal_Pasta or Vegetarian')]


df1 = (pd.melt(temp1,id_vars = list_melt, value_name='Meal'))

df1[['tmp','cat']] = df1.variable.str.split('_', expand=True)

df1 = df1.drop(['variable', 'tmp'],axis=1).sort_values(['flight_number', 'flight_boarding_pt', 'flight_boarding_time'])

df1 = df1[df1.Meal == 1]


pax_count = M1[['flight_number', 'flight_boarding_pt', 
                           'flight_boarding_time','pax_id']].groupby(['flight_number', 'flight_boarding_pt', 
                           'flight_boarding_time']).agg({'pax_id':'nunique'}).reset_index()
pax_count.columns = ['flight_number', 'flight_boarding_pt', 'flight_boarding_time', 'pax_count']
#pax_count.pax_count = pax_count.pax_count/3


df1 = pd.merge(df1,pax_count,on=['flight_number', 'flight_boarding_pt', 'flight_boarding_time'])

df1.rename(columns = {'cat':'dishsubcategory','service_category_code' : 'itemcategory'}, inplace = True)

df1.itemcategory = df1.itemcategory.replace({'L':'Lunch','D':'Dinner'})

#df1.to_csv('MVP3_testing_Future_data.csv',index=False)
#df1.to_csv('MVP3_testing_Future_data_June7.csv',index=False)
df1.to_csv('MVP3_testing_Future_data_June_21_27_28.csv',index=False)
# combine and time series related variables


#zz1 = zz[['flight_number', 'flight_boarding_pt', 
#                           'flight_boarding_time']].groupby(['flight_number', 'flight_boarding_pt']).max().reset_index()
#
#zz2 = zz[['flight_number', 'flight_boarding_pt', 
#                           'flight_boarding_time']].groupby(['flight_number', 'flight_boarding_pt']).min().reset_index()
