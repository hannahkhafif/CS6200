# data set cleaning, feature engineering, feature selection
# imputation, one hot label encoding, lasso regression, scaling

#import libraries
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings('ignore')


# read in the data
data = pd.read_csv('home-credit-default-risk/application_train.csv')
print("data read into df")


#********
# label encoding for categorical columns w/ only 2 categories
le= LabelEncoder()

#iterate through columns
for col in data:
    # If categorical data and 2 or fewer unique categories 
    if data[col].dtype == 'object' and len(list(data[col].unique())) <= 2:
        data[col] = le.fit_transform(data[col])


#***********
#one-hot encoding for rest of categorical cols
data= pd.get_dummies(data)
print("one hot encoding complete")


#fix outliers in data
data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])

#impute missing data
data= data.apply(lambda x: x.fillna(x.mean()), axis=0)
print("imputation done")

# save cleaned full dataset to csv
data.to_csv('cleandata.csv', index=False)   


#***********************************************************************
print("Feature engineering extra features")
# 
data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
#----------------------------------------------------------------------
#***********************************************************************
# Feature engineering: getting info from other data files
#***********************************************************************
print("Feature engineering: data merging")
# burea and burea_balance
bureau= pd.read_csv('home-credit-default-risk/bureau.csv')

b= bureau.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM_DEBT': 'sum'})
b['TOTAL_OVERDUE']=bureau.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM_OVERDUE':'sum'})
b['NUM_PROLONG']=bureau.groupby('SK_ID_CURR').agg({'CNT_CREDIT_PROLONG':'sum'})
b['TOTAL_ANNUITY']=bureau.groupby('SK_ID_CURR').agg({'AMT_ANNUITY':'sum'}) 
b[['MEAN_DAYS_APPLY','MAX_DAYS_APPLY']]=bureau.groupby('SK_ID_CURR').agg({'DAYS_CREDIT':['mean','max']})
b[['MEAN_DAYS_OVERDUE','MAX_DAYS_OVERDUE']]=bureau.groupby('SK_ID_CURR').agg({'CREDIT_DAY_OVERDUE':['mean','max']})

b['NUM_ACTIVE_LOANS']=bureau[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR').count()['CREDIT_ACTIVE']
b['NUM_IN_YEAR']= bureau[bureau.DAYS_CREDIT>=-365].groupby('SK_ID_CURR').count()['DAYS_CREDIT']



new_data= pd.merge(data, b, how='left', on='SK_ID_CURR')
print("saving full data set to csv")
new_data.to_csv('fulldata.csv', index=False)  


#----------------------------------------------------------------------
#imputation of new null values
new_data['AMT_CREDIT_SUM_DEBT'].replace({np.nan:0}, inplace=True)
new_data['TOTAL_OVERDUE'].replace({np.nan:0}, inplace=True)
new_data['NUM_PROLONG'].replace({np.nan:0}, inplace=True)
new_data['TOTAL_ANNUITY'].replace({np.nan:0}, inplace=True)
new_data['MEAN_DAYS_APPLY'].replace({np.nan:0}, inplace=True)
new_data['MAX_DAYS_APPLY'].replace({np.nan:0}, inplace=True)
new_data['MEAN_DAYS_OVERDUE'].replace({np.nan:0}, inplace=True)
new_data['MAX_DAYS_OVERDUE'].replace({np.nan:0}, inplace=True)
new_data['NUM_ACTIVE_LOANS'].replace({np.nan:0}, inplace=True)
new_data['NUM_IN_YEAR'].replace({np.nan:0}, inplace=True)


#----------------------------------------------------------------------
#   feature selection with lasso regression

X_train, X_test, y_train, y_test = train_test_split(
    new_data.drop(labels=['TARGET', 'SK_ID_CURR'], axis=1),
    new_data['TARGET'],
    test_size=0.3,
    random_state=0)


print("scaling data")
# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)

print("Starting lasso regression for feature selection")
selector = SelectFromModel(LogisticRegression(C=1, solver='saga', penalty='l1'), max_features=150)
selector.fit(scaler.transform(X_train), y_train)

selected_feat = X_train.columns[(selector.get_support())]
print("Lasso Regression Complete")

#--------------------------------------------------------------------------
sel= new_data[selected_feat]
sel['TARGET']= new_data['TARGET']
sel['SK_ID_CURR']= new_data['SK_ID_CURR']

#-------------------------------------------------------------------------
print("saving selected features to csv")
#dataframe to csv
sel.to_csv('home_credit_data.csv', index=False)

