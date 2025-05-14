import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import multiprocessing as mp
import gc
import datetime
from sklearn.preprocessing import LabelEncoder
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

features = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/features.csv")
stores = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/stores.csv")
train = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/train.csv")
test = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/walmart-sales-forecast/test.csv")

#------------------------------------------------------------------
# constants
#------------------------------------------------------------------

CORR_THRESHOLD = 0.1

#------------------------------------------------------------------
# df
#------------------------------------------------------------------



#------------------------------------------------------------------
# uncorrelated
#------------------------------------------------------------------
def removeUnCorrelatedCols(df, threshold=CORR_THRESHOLD):
    rst = []
    num_columns = len(df.columns)
    corr_matrix = df.corr()

    for ci in corr_matrix.columns:
        list0 = []
        for cj in corr_matrix.columns:
            if (abs(corr_matrix[ci][cj]) < threshold and ci != cj):
                list0.append(cj)

        if len(list0) == num_columns - 1:
            rst.append(ci)

    rst = np.array(rst)
    to_drop = np.unique(rst)
    df.drop(to_drop, axis=1, inplace=True)
    return to_drop


#------------------------------------------------------------------
# outliers
#------------------------------------------------------------------

def getOutliersIqr(df, column_name, threshold=1.5):
    q1 = df[column_name].quantile(0.25)

    q3 = df[column_name].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = df[(df[column_name] >= upper_bound) | (df[column_name] <= lower_bound)]
    return outliers


def getOutliersZscore(df, column_name, threshold=3):
    z = np.abs(stats.zscore(df[column_name]))
    print("z = ", z)
    print("len(z) = ", len(z))
    outliers = df.loc[z >= threshold]
    return outliers

def getIsna(df, column_name):
    z = df[column_name].isna()
    return z

def filterNonNumericalColV1(df):
    df_numeric = df.select_dtypes(include=['number'])
    return df_numeric

def filterOutlierV0(x, col_name, low, high):
    if x[col_name] < low or x[col_name] > high:
        return True
    return False

def removeOutlierV0(df_in, col_name, low_threshold=0.25, high_threshold=0.75, threshold=1.5):
    q1 = df_in[col_name].quantile(low_threshold)
    q3 = df_in[col_name].quantile(low_threshold)

    iqr = q3 - q1
    low = q1 - threshold * iqr
    high = q3 + threshold * iqr

    mask = df_in.apply(filterOutlierV0, axis=1, args=(col_name, low, high))
    print("mask = ")
    print(mask)

    df_in.loc[mask, col_name] = df_in[col_name].median()
    return

def filterOutlierV1(x, col_name, mean_val, std_val, threshold=3.0):
    if np.abs(x[col_name] - mean_val) > threshold * std_val:
        return True
    return False


def removeOutlierV1(df_in, col_name, threshold=3.0):
    mean_val = df_in[col_name].mean()
    std_val = df_in[col_name].std()

    mask=df_in.apply(lambda row: filterOutlierV1(row, col_name, mean_val, std_val, threshold), axis=1)
    print("mask = ")
    print(mask)

    df_in.loc[mask, col_name] = df_in[col_name].median()
    return




