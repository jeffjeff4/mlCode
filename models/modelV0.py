import sys
print(sys.path)
sys. path.append("//Users//shizhefu0//Desktop//ml//code//github_jeffjeff4//")

import pandasHelperV0

import string

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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



df_features = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/features.csv")
df_stores = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/stores.csv")
df_train = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/train.csv")
df_test = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/test.csv")

#------------------------------------------------------------------
# constants
#------------------------------------------------------------------

CORR_THRESHOLD = 0.1

#------------------------------------------------------------------
# code
#------------------------------------------------------------------

df_features.head()
df_features.info()

df_stores.head()
df_stores.info()

df_train.head()
df_train.info()

df_test.head()
df_test.info()

df_features_stores = df_features.merge(df_stores, how="inner", on=['Store'])
df_features_stores['Date'] = pd.to_datetime(df_features_stores["Date"])
pandasHelperV0.getDatasetInfo(df_features_stores, "df_features_stores", need_plot=False)

df_features_stores['Day'] = df_features_stores['Date'].dt.isocalendar().day
df_features_stores['Week'] = df_features_stores['Date'].dt.isocalendar().week
df_features_stores['Month'] = df_features_stores['Date'].dt.isocalendar().month
df_features_stores['Year'] = df_features_stores['Date'].dt.isocalendar().year

col_name = 'CPI'
pandasHelperV0.drawHistogram(df_features_stores, col_name)




