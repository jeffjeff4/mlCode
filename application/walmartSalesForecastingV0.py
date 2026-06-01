import sys
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

le = LabelEncoder()
df_train['IsHoliday'] = le.fit_transform(df_train['IsHoliday'])
df_train['Type'] = le.fit_transform(df_train['Type'])

df_train['Week'] = df_train['Week'].astype(int)
df_train['CPI'] = df_train['CPI'].astype(int)

pdh.getDfInfo(df_train, 'df_train')

numerical_cols = df_train.select_dtypes(include=['number']).columns.tolist()
num_rows_before = len(df_train)
print("num_rows_before = ", num_rows_before)

for col in numerical_cols:
    pdh.removeOutlierV0(df_train, col)

num_rows_after = len(df_train)
print("num_rows_after = ", num_rows_after)
