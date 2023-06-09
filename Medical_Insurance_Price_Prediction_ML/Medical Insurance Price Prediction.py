import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import warnings
warnings.filterwarnings('ignore')

# Data Import
df = pd.read_csv('insurance.csv')
#print(f'Shape: {df.shape}')
#print(f'Dtypes: {df.dtypes}')
#print(f'DF_Info: {df.info()}')


# EDA
features = ['sex', 'smoker', 'region', 'children']

#plt.subplots(figsize=(20, 10))
#for i, col in enumerate(features):
#    plt.subplot(1, 4, i + 1)
#
#    x = df[col].value_counts()
#    plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
#plt.show()

features = ['age', 'bmi']

#plt.subplots(figsize= (17, 7))
#for i, col in enumerate(features):
#    plt.subplot(1, 2, i + 1)
#    sb.scatterplot(data=df, x=col, y='expenses', hue='smoker')
#plt.show()

#plt.subplots(figsize= (17, 7))
#for i, col in enumerate(features):
#    plt.subplot(1, 2, i + 1)
#    sb.distplot(df[col])
#plt.show()

#plt.subplots(figsize=(15,5))
#for i, col in enumerate(features):
#    plt.subplot(1, 2, i + 1)
#    sb.boxplot(df[col])
#plt.show()



# Data Transforming
df = df[df['bmi']<45]

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


#plt.figure(figsize=(7,7))
#sb.heatmap(df.corr(), annot=True, cbar=False)
#plt.show()



# Model Training
features = df.drop('expenses', axis=1)
target = df['expenses']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.25, random_state=22)
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_train)

#X_train = X_train.transpose()
#X_val = X_val.transpose()

models = [LinearRegression(), XGBRFRegressor(), RandomForestRegressor(), AdaBoostRegressor(), Lasso(), Ridge()]

for i in range(6):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    pred_train = models[i].predict(X_train)
    print(f'Train Accuracy : {1 - (mape(Y_train, pred_train))}')

    val_train = models[i].predict(X_val)
    print(f'Validation Accuracy : {1 - (mape(Y_val, val_train))}')
    print('|')