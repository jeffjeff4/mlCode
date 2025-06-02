import sys

from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor
from tensorflow import double
from tensorflow.python.feature_column.feature_column import linear_model
from tensorflow.python.ops.numpy_ops.np_dtypes import float64

#from walmartSalesForecastingV0 import num_rows_before, numerical_cols

sys.path.append("//Users//shizhefu0//Desktop//ml//code//github_jeffjeff4")
print(sys.path)
import pandasHelperV0 as pdh

import string
from lib2to3.btm_utils import tokens

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import pad_sequence
from sklearn.cluster import KMeans
import multiprocessing as mp
import gc
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import calendar
from scipy.sparse import csr_matrix,hstack
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from tensorflow.python.ops.gen_array_ops import upper_bound
#from lightgbm import LGBMRegressor
from tqdm import tqdm
import pickle
from scipy import stats

import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

import string

from collections import Counter
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
#import catboost as cb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

models = {
    'lr' : linear_model.LinearRegression(),
    'xbg' : xgb.XGBRegressor(random_state=1, objective='reg:squarederror'),
#    'cb' : cb.LGBMRegressor(random_state=1),
    'rfr' : RandomForestRegressor(random_state=1)
}


OUTLIER_THRESHOLD = 3.0
BATCH_SIZE = 32
EMBEDDING_DIM = 64
NUM_HEADS = 4
MODEL_DIM = 128
NUM_LAYERS = 4
OUTPUT_DIM = 1

df_features = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/features.csv")
df_stores = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/stores.csv")
train = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/train.csv")
test = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/test.csv")

pdh.getDfInfo(df_features, "features")
pdh.getDfInfo(df_stores, "stores")
pdh.getDfInfo(train, "train")
pdh.getDfInfo(test, "test")

################################
# data processing
################################

df_feature_stores = df_features.merge(df_stores, how='inner', on=['Store'])
df_feature_stores['Date'] = pd.to_datetime(df_feature_stores['Date'])

df_feature_stores['Day'] = df_feature_stores['Date'].dt.isocalendar().day
df_feature_stores['Week'] = df_feature_stores['Date'].dt.isocalendar().week
df_feature_stores['Month'] = df_feature_stores['Date'].dt.month
df_feature_stores['Year'] = df_feature_stores['Date'].dt.isocalendar().year

pdh.getDfInfo(df_feature_stores, "df_feature_stores", need_plot=False)

#col_name = "CPI"
#pdh.drawHistogram(df_feature_stores, col_name)

col_name_x = "Store"
col_name_y = "CPI"
pdh.drawScatterplot(df_feature_stores, col_name_x, col_name_y)

col_name_x = "Store"
col_name_y = "Temperature"
pdh.drawScatterplot(df_feature_stores, col_name_x, col_name_y)

col_name_x = "Week"
col_name_y = "CPI"
pdh.drawScatterplot(df_feature_stores, col_name_x, col_name_y)

col_name_x = "Week"
col_name_y = "Temperature"
pdh.drawScatterplot(df_feature_stores, col_name_x, col_name_y)

################################
# training data
################################

train['Date'] = pd.to_datetime(train['Date'])
df_train = train.merge(df_feature_stores, how='inner', on=['Store', 'Date', 'IsHoliday'])

column_name = 'Date'
df_train['Year'] = df_train[column_name].dt.isocalendar().year
df_train['Day'] = df_train[column_name].dt.isocalendar().day
df_train['Week'] = df_train[column_name].dt.isocalendar().week
df_train['Month'] = df_train[column_name].dt.month

pdh.getDfInfo(df_train, "df_train", need_plot=False)

le = LabelEncoder()
df_train['IsHoliday'] = le.fit_transform(df_train['IsHoliday'])
df_train['Type'] = le.fit_transform(df_train['Type'])

df_train['Week'] = df_train['Week'].astype(int)
df_train['CPI'] = df_train['CPI'].astype(int)

pdh.getDfInfo(df_train, "df_train")

print()
num_rows_before_remove_outlier_df_train = len(df_train)
numerical_cols = df_train.select_dtypes(include=['number']).columns.tolist()

for col in numerical_cols:
    z = np.abs(stats.zscore(df_train[col]))
    mask = df_train[z>OUTLIER_THRESHOLD]
    df_train = df_train[~(df_train.index.isin(mask.index))]

num_rows_after_remove_outlier_df_train = len(df_train)
print("------------------------training data----------------------------")
print("training data, before remove outlier, num rows = ", num_rows_before_remove_outlier_df_train,
      ", after remove outlier, num rows = ", num_rows_after_remove_outlier_df_train)
print()

def filterOutlierV1(df, col, mean_val, std_val, threshold=OUTLIER_THRESHOLD):
    if (np.abs(df[col] - mean_val) > std_val * threshold):
        return True
    return False

#for col in numerical_cols:
#    mean_val = df_train[col].mean()
#    std_val = df_train[col].std()
#    mask = df_train.apply(lambda row: pdh.filterOutlierV1(df_train, col, mean_val, std_val, OUTLIER_THRESHOLD), axis=1)
#    df_train = df_train[~mask]

num_rows_after_remove_outlier_df_train1 = len(df_train)
print("training data, before remove outlier1, num rows = ", num_rows_after_remove_outlier_df_train,
      ", after remove outlier1, num rows = ", num_rows_after_remove_outlier_df_train1)
print()

print()
print("before, df_train.isna().sum() = ")
print(df_train.isna().sum())

num_rows_before_remove_na_df_train = len(df_train)
for col in numerical_cols:
    mask = df_train[col].isna()
    df_train = df_train[~mask]

num_rows_after_remove_na_df_train = len(df_train)

print("------------------------training data----------------------------")
print("training data, before remove na, num rows = ", num_rows_before_remove_na_df_train,
      ", after remove na, num rows = ", num_rows_after_remove_na_df_train)
print()

print()
print("df_train.corr = ")
print(df_train.corr(method="spearman", numeric_only=True))

################################
# testing data
################################

pdh.getDfInfo(test, "test")

df_test = train.merge(df_feature_stores, how='inner', on=['Store', 'Date', 'IsHoliday'])

column_name = 'Date'
df_test['year'] = df_test[column_name].dt.isocalendar().year
df_test['Day'] = df_test[column_name].dt.isocalendar().day
df_test['Week'] = df_test[column_name].dt.isocalendar().week
df_test['Month'] = df_test[column_name].dt.month

pdh.getDfInfo(df_test, "df_test", need_plot=False)

le = LabelEncoder()
df_test['IsHoliday'] = le.fit_transform(df_test['IsHoliday'])
df_test['Type'] = le.fit_transform(df_test['Type'])

df_test['Week'] = df_test['Week'].astype(int)
df_test['CPI'] = df_test['CPI'].astype(int)

pdh.getDfInfo(df_test, "df_test")

print()
print("before, df_test.isna().sum() = ")
print(df_test.isna().sum())

num_rows_before_remove_na_df_test = len(df_test)
for col in numerical_cols:
    mask = df_test[col].isna()
    df_test = df_test[~mask]

num_rows_after_remove_na_df_test = len(df_test)

print("------------------------testing data----------------------------")
print("testing data, before remove na, num rows = ", num_rows_before_remove_na_df_test,
      ", after remove na, num rows = ", num_rows_after_remove_na_df_test)
print()

#-------------------------------------
#training
#-------------------------------------
target_col = 'Weekly_Sales'

col_name_x = 'Store'
col_name_y = target_col
pdh.drawScatterplot(df_train, col_name_x, col_name_y)

col_name_x = 'Temperature'
col_name_y = target_col
pdh.drawScatterplot(df_train, col_name_x, col_name_y)

col_name_x = 'CPI'
col_name_y = target_col
pdh.drawScatterplot(df_train, col_name_x, col_name_y)

col_name_x = 'Week'
col_name_y = target_col
pdh.drawScatterplot(df_train, col_name_x, col_name_y)

df_test['IsHoliday'] = le.fit_transform(df_test['IsHoliday'])
df_test['Type'] = le.fit_transform(df_test['Type'])

df_test['Week'] = df_test['Week'].astype(int)
df_test['CPI'] = df_test['CPI'].astype(float)

features = ["Week", "CPI", "Unemployment", "Size", "Type", "Dept", "Store"]

from sklearn.model_selection import train_test_split
X = df_train[features].copy()
y = df_train[target_col].copy()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, test_size=0.2)

print("-------------------------")
print("type(X_train) = ", type(X_train))
print("type(y_train) = ", type(y_train))
print("X_train.shape = ", X_train.shape)
print("y_train.shape = ", y_train.shape)

print("type(X_valid) = ", type(X_valid))
print("type(y_valid) = ", type(y_valid))
print("X_valid.shape = ", X_valid.shape)
print("y_valid.shape = ", y_valid.shape)

test_x = df_test[features].copy()

print("----------------------------------------------------")
print("training validations")
print("----------------------------------------------------")

def validModel(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    preds = model.predict(X_valid.to_numpy())
    rmse = mean_squared_error(y_valid.to_numpy(), preds)

    return rmse

#for name, model in models.items():
#    train_valid_rmse = validModel(model, X_train, y_train, X_valid, y_valid)
#    print("train validation {0} : {1}".format(name, train_valid_rmse))

#    test_preds = model.predict(test_x.to_numpy())
#    print()

print("----------------------------------------------------")
print("testing validations")
print("----------------------------------------------------")

#-----------------------------------------
#dl model
#-----------------------------------------

df_train = df_train.select_dtypes(include='number')
df_train = df_train.astype('double')
#df_train = df_train.astype('float')

tmp_x = df_train[features].copy()
tmp_y = df_train[target_col].copy()

x_np = tmp_x.to_numpy()
y_np = tmp_y.to_numpy()

x_train = torch.tensor(x_np)
y_train = torch.tensor(y_np)

tmp_x_1 = df_test[features].copy()
#tmp_y_1 = df_test[target_col].copy

x_np_1 = tmp_x_1.to_numpy()
#y_np_1 = tmp_y_1.to_numpy()

x_test = torch.tensor(x_np_1)
#y_test = torch.tensor(y_np_1)


class NumericEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.linear(x)


def train(model, optimizer, criterion, dataLoader, epochs=10):
    for epoch in range(epochs):
        model.train()
        ttl_loss = 0.0
        for x_batch, y_batch in dataLoader:
            optimizer.zero_grad()
            output = model(x_batch)

            #loss = nn.MSELoss()(output, y_batch)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

            loss = loss.detach()
            ttl_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, loss: {ttl_loss/len(dataLoader):.4f}")


def predict(model, input):
    model.eval()
    with torch.no_grad:
        rst = model(input)
        return rst

dataset = TensorDataset(x_train, y_train)

embedding_layer = NumericEmbedding(len(features), EMBEDDING_DIM)
embedding_layer = embedding_layer.double()

dl_regressor = pdh.TransformerRegressor(EMBEDDING_DIM, MODEL_DIM, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM)
dl_regressor = dl_regressor.double()

# We'll train them end-to-end
model = nn.Sequential(embedding_layer, dl_regressor)

dataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


train(model, optimizer, criterion, dataLoader)
test_preds = predict(model, x_test)

print(f"test preds dl_regressor : {test_preds}")
print()

print("df_train.head(10):")
print(df_train.head(10))
df_train_np = df_train.to_numpy()
#df_train_np = df_train_np.astype(float)
df_train_tensor = torch.tensor(df_train_np)
