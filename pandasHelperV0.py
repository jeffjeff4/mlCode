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
import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


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
def getDfInfo(df, df_name, need_plot=False):
    print("len_df = ", len(df))
    print(df.head())
    print("{0}.head = ".format(df_name))
    print(df.head(10))
    print("{0}.info = ".format(df_name))
    print(df.info())
    print("{0}.describe = ".format(df_name))
    print(df.describe())
    if need_plot==True:
        df.plot()
        plt.title("{0}.info = ".format(df_name))
        plt.show()


##import pandas as pd
##
### Sample DataFrame
##df = pd.DataFrame({
##    'A': [1, 2, 3],
##    'B': ['x', 'y', 'z'],
##    'C': [True, False, True]
##})
##
##print('df')
##print(df)
##print('\n')
##
### Drop single column (e.g., 'B')
##df_dropped = df.drop('B', axis=1)
##print('df_dropped')
##print(df_dropped)
##print('\n')
##
### Drop the second row (index=1)
##df_dropped = df.drop(index=df.index[1])
##print('df_dropped, 111')
##print(df_dropped)
##print('\n')
##
### Modify the DataFrame directly (no copy)
##df.drop(['A', 'C'], axis=1, inplace=True)
##print('df after dropped')
##print(df)
##print('\n')
##
##
### Sample DataFrame
##df = pd.DataFrame({
##    'A': [1, 2, 3, 4, 5, 6],
##    'B': ['x', 'y', 'z', 'a', 'b', 'c'],
##    'C': [True, False, True, False, True, False]
##})
##
### Drop rows 'row1' and 'row3'
##df_dropped = df.drop(index=df.index[[1, 2]])
##print(df_dropped)
##
##print('df_dropped, 222')
##print(df_dropped)
##print('\n')
##
##
### Drop rows where column 'A' > 1
##df_dropped = df[df['A'] <= 1]
##print('df_dropped, 333')
##print(df_dropped)
##print('\n')
##
##
### Drop rows where column 'A' > 1
##df_dropped = df.drop(df[df['A'] >= 6].index)
##print('df_dropped, 444')
##print(df_dropped)
##print('\n')

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


##import pandas as pd
##from sklearn.model_selection import train_test_split
##from sklearn.preprocessing import StandardScaler
##
### 1. Load data
##data = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/##deepseek_csv_v0.txt")
##print("data = ")
##print(data)
##print('\n')
##
####age,income,gender,purchase_history,target
####25,50000,Male,3,1
####30,80000,Female,5,1
####22,30000,Male,1,0
####35,120000,Female,7,1
####28,45000,Male,2,0
####40,95000,Female,8,1
####19,20000,Male,0,0
####33,110000,Female,6,1
####26,40000,Male,1,0
####45,150000,Female,10,1
####19,20000,Male,0,0
####28,45000,Male,2,0
####26,40000,Male,1,0
##
#--------------------------------------------------------
# drop duplicate
#--------------------------------------------------------

### 2. Clean data
##data.drop_duplicates(inplace=True)
##
##print('data.drop_duplicates')
##print(data)
##print('\n')
##

###-------------------------------------------------
### data format conversion
###-------------------------------------------------

##import pandas as pd
##
### Sample data with date strings
##data = {'date_str': ['2023-01-15', '2023-02-20', '2023-03-25']}
##df = pd.DataFrame(data)
##
### Convert string to datetime
##df['date'] = pd.to_datetime(df['date_str'])
##print(df.dtypes)
##
##df['numeric'] = pd.to_numeric(df['string_column'], errors='coerce')
##
##df['string'] = df['numeric_column'].astype(str)
##
##df['bool'] = df['string_column'].map({'true': True, 'false': False})
##
##df['category'] = df['string_column'].astype('category')
##
##df['year'] = df['date'].dt.year
##df['month'] = df['date'].dt.month
##df['day'] = df['date'].dt.day
##df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6

###-------------------------------------------------
### column operation
###-------------------------------------------------
##
##print("#-------------------------------------------------")
##print('column operation')
##print("#-------------------------------------------------")
##
##import pandas as pd
##
### Sample DataFrame
###df = pd.DataFrame({
###    'A': [1, 2, 3],
###    'B': ['x', 'y', 'z'],
###    'C': [True, False, True]
###})
##
##df = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/##deepseek_csv_v0.txt")
##
##print('df original')
##print(df)
##print('\n')
##
### Loop through columns
##for column in df.columns:
##    print(f"Column: {column}")
##    print(f"Data type: {df[column].dtype}")
##    print(f"Values:\n{df[column]}\n")
##
### Example: Check if each column has missing values
##print("df.apply(lambda col: col.isnull().sum())")
##print(df.apply(lambda col: col.isnull().sum()))
##print('\n')
##
### Get the first column (index 0)
##first_column = df.iloc[:, 0]
##print("first_column")
##print(first_column)
##print('\n')
##
##print("df Mean (if numeric)")
##for column_name, column_data in df.items():
##    print(f"Column: {column_name}")
##    print(f"Mean (if numeric): {column_data.mean() if ##pd.api.types.is_numeric_dtype(column_data) else 'N/A'}")
##print('\n')
##
### Convert all strings to uppercase
##for column in df.columns:
##    if pd.api.types.is_string_dtype(df[column]):
##        df[column] = df[column].str.upper()
##print("Convert all strings to uppercase")
##print(df)
##print('\n')
##
##
### Example: Standardize all numeric columns
##df_numeric = df.select_dtypes(include='number')
##df[df_numeric.columns] = (df_numeric - df_numeric.mean()) / ##df_numeric.std()
##print("Standardize all numeric columns")
##print(df)
##print('\n')
#
#--------------------------------------------------------
# go through each column, and replacing na with median
#--------------------------------------------------------
#
##for column in df.columns:
##    print("df[column].dtype = ", df[column].dtype)
##    if df[column].dtype in ['number', 'int8', 'int64', ##'float32', 'float16']:
##        df.fillna(df[column].median(), inplace=True)
##
##print('df after repalcing na with median')
##print(df)
##print('\n')

##from pandas.api.types import is_numeric_dtype, is_string_dtype, ##is_datetime64_dtype
##
### Check if a column is numeric
##print("is_numeric_dtype(df[\'age\'])")
##print(is_numeric_dtype(df['age']))  # True
##print('\n')
##
### Check if a column is string (object)
##print("is_string_dtype(df[\'gender\'])")
##print(is_string_dtype(df['gender']))   # True
##print('\n')
##
### Check if a column is datetime
##print("is_datetime64_dtype(df[\'target\'])")
##print(is_datetime64_dtype(df['target']))  # True
##print('\n')
##
### Convert 'A' to float
##df['age'] = df['age'].astype('float64')
##print("df[\'age\'].dtype")
##print(df['age'].dtype)  # Output: float64
##print('\n')
##
### Convert 'B' to categorical
##df['gender'] = df['gender'].astype('category')
##print("df[\'gender\'].dtype")
##print(df['gender'].dtype)  # Output: category
##print('\n')

#****************************
#manual code
#****************************

#col_name = 'Rating'
#new_col_name = 'Class'
#training_df = review_clean_df
#training_df.loc[training_df[col_name].isin([1.0, 2.0]), new_col_name] = 'Bad'
#training_df.loc[training_df[col_name].isin([3.0]), new_col_name] = 'Neutral'
#training_df.loc[training_df[col_name].isin([4.0, 5.0]), new_col_name] = 'Good'


#--------------------------------------------------------
# non numerical feature encoding
#--------------------------------------------------------

#*****************************
#example
#df = pd.DataFrame({'city': ['ny', 'la', 'sf', 'la']})
#df_encoded = pd.get_dummies(df, columns=['city'], drop_first=True)
#print(df_encoded)

#usage:
#df_encoded_sklearn_dropped = pd.concat([df, encoded_df], axis=1).drop('color', axis=1)

#sample to call this function
#col_name = ['color']
#test_df0 = pdh.oneHotEncoderPd(df, col_name)
#test_df0.head()
#*****************************

def oneHotEncoderPd(df, col_name, is_drop_first=True):
    df_encoded = pd.get_dummies(df, columns=col_name, drop_first=is_drop_first)
    return df_encoded

#*****************************
#example
# sample categorical data (Numpy array or Pandas DataFrame)
# data = np.array([['red'], ['blue'], ['green'], ['red'], ['yellow']])
# or as pandas dataframe:
#  df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'yellow']})

#usage
# df_encoded_sklearn_dropped = pd.concat([df, encoded_df], axis=1).drop('color', axis=1)

#sample to call this function
#col_name = ['color']
# df_encoded = pandasHelpers.oneHotEncoderSklearn(df, col_name=col_name, is_drop_first=True)
# df_encoded.head()
#*****************************

def oneHotEncoderSklearn(df, col_name):
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    encoded_array_df = encoder.fit_transform(df[col_name])
    feature_names = encoder.get_feature_names_out(col_name)

    df_encoded = pd.DataFrame(encoded_array_df, columns=feature_names)
    return df_encoded

from sklearn.preprocessing import LabelEncoder
def featureEncode(df, col_name):
    le = LabelEncoder()
    new_col_name = col_name + "_encoded"
    df[new_col_name] = le.fit_transform(df[col_name])
    return df


##import pandas as pd
##from sklearn.preprocessing import OneHotEncoder
##
### Sample data
##data = pd.DataFrame({
##    'color': ['red', 'blue', 'green', 'blue', 'red'],
##    'size': ['S', 'M', 'L', 'M', 'XL']
##})
##
### One-hot encode using pandas
##onehot_pd = pd.get_dummies(data, columns=['color', 'size'])
##print("Pandas one-hot:\n", onehot_pd.head())
##
### Using scikit-learn
##encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first avoids multicollinearity
##onehot_sk = encoder.fit_transform(data[['color', 'size']])
##print("\nScikit-learn one-hot:\n", onehot_sk[:3])
##
##from sklearn.preprocessing import LabelEncoder
##
### For ordinal data with meaningful order
##size_order = ['XS', 'S', 'M', 'L', 'XL']
##data['size_encoded'] = data['size'].apply(lambda x: size_order.index(x))
##print("\nManual label encoding:\n", data[['size', 'size_encoded']].head())
##
### Using scikit-learn (caution: doesn't preserve order)
##le = LabelEncoder()
##data['color_encoded'] = le.fit_transform(data['color'])
##print("\nLabelEncoder output:\n", data[['color', 'color_encoded']].head())
##
##from category_encoders import TargetEncoder
##from sklearn.model_selection import train_test_split
##
### Create target variable
##data['target'] = [1, 0, 1, 1, 0]
##
### Target encoding with regularization
##X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)
##encoder = TargetEncoder(cols=['color'])
##encoder.fit(X_train['color'], X_train['target'])
##data['color_encoded'] = encoder.transform(data['color'])
##print("\nTarget encoding:\n", data[['color', 'color_encoded']].head())
##
### Count frequency of each category
##freq_map = data['color'].value_counts(normalize=True)
##data['color_freq'] = data['color'].map(freq_map)
##print("\nFrequency encoding:\n", data[['color', 'color_freq']].head())
##
##from category_encoders import BinaryEncoder
##
##encoder = BinaryEncoder(cols=['color'])
##binary_encoded = encoder.fit_transform(data['color'])
##print("\nBinary encoding:\n", binary_encoded.head())
##
##from sklearn.feature_extraction import FeatureHasher
##
##hasher = FeatureHasher(n_features=4, input_type='string')
##hashed = hasher.transform(data['color'].apply(lambda x: [x])).toarray()
##hashed_df = pd.DataFrame(hashed, columns=[f'color_hash_{i}' for i in range(4)])
##print("\nFeature hashing:\n", hashed_df.head())
##
##from tensorflow.keras.layers import Embedding, Input
##from tensorflow.keras.models import Model
##import numpy as np
##
### Prepare data
##categories = data['color'].unique()
##cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
##data['color_idx'] = data['color'].map(cat_to_idx)
##
### Create embedding model
##input_layer = Input(shape=(1,))
##embedding = Embedding(input_dim=len(categories), output_dim=2)(input_layer)
##model = Model(inputs=input_layer, outputs=embedding)
##
### Get embeddings
##embeddings = model.predict(np.array(data['color_idx']))
##print("\nEmbedding vectors:\n", embeddings[:3])
##
### Group rare categories into 'other'
##threshold = 0.1  # Minimum frequency to keep
##freq = data['color'].value_counts(normalize=True)
##data['color_clean'] = data['color'].where(data['color'].isin(freq[freq > threshold].index), 'other')
##print("\nHandling rare categories:\n", data['color_clean'].value_counts())
##
##from sklearn.compose import ColumnTransformer
##from sklearn.pipeline import Pipeline
##from sklearn.ensemble import RandomForestClassifier
##
### Define preprocessing
##preprocessor = ColumnTransformer(
##    transformers=[
##        ('onehot', OneHotEncoder(), ['color']),
##        ('target', TargetEncoder(), ['size'])
##    ],
##    remainder='passthrough'
##)
##
### Create pipeline
##pipeline = Pipeline([
##    ('preprocessor', preprocessor),
##    ('classifier', RandomForestClassifier())
##])
##
### Example usage (X_train, y_train would be your actual data)
### pipeline.fit(X_train, y_train)
##
##from sklearn.model_selection import cross_val_score
##
### Compare encodings
##encoders = {
##    'OneHot': OneHotEncoder(),
##    'Target': TargetEncoder(),
##    'Binary': BinaryEncoder()
##}
##
##for name, encoder in encoders.items():
##    preprocessor = ColumnTransformer(
##        [('encoder', encoder, ['color', 'size'])],
##        remainder='drop'
##    )
##    X_encoded = preprocessor.fit_transform(data, data['target'])
##    scores = cross_val_score(RandomForestClassifier(), X_encoded, data['target'], cv=3)
##    print(f"{name} encoding accuracy: {scores.mean():.3f}")

#----------------------------------------------
#predictive impute
#----------------------------------------------

##from sklearn.impute import KNNImputer
##import pandas as pd
##import numpy as np
##
### Create sample data with missing values
##data = {'Age': [25, 30, np.nan, 35, 40],
##        'Income': [50000, np.nan, 70000, np.nan, 90000],
##        'Education': [12, 16, np.nan, 14, 18]}
##df = pd.DataFrame(data)
##
##print('original df')
##print(df)
##print('\n')
##
### Initialize KNN imputer
##imputer = KNNImputer(n_neighbors=2)
##
### Perform imputation
##df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
##print('knn df_imputed')
##print(df_imputed)
##print('\n')
##
##from lightgbm import LGBMRegressor, LGBMClassifier
##from sklearn.model_selection import train_test_split
##
##
##def predictive_impute(df, target_col):
##    # Split into missing and non-missing
##    missing = df[df[target_col].isna()]
##    not_missing = df[~df[target_col].isna()]
##
##    # Prepare features and target
##    X = not_missing.drop(target_col, axis=1)
##    y = not_missing[target_col]
##
##    # Determine if classification or regression
##    if y.dtype == 'object' or len(y.unique()) < 10:
##        model = LGBMClassifier()
##    else:
##        model = LGBMRegressor()
##
##    # Train model
##    model.fit(X, y)
##
##    # Predict missing values
##    X_missing = missing.drop(target_col, axis=1)
##    df.loc[df[target_col].isna(), target_col] = model.predict(X_missing)
##
##    return df
##
##
### Example usage
##df = predictive_impute(df, 'Age')
##print('lightgbm df_imputed')
##print(df)
##print('\n')


#--------------------------------------------------------
# data transformation
#--------------------------------------------------------

##from sklearn.preprocessing import StandardScaler
##import numpy as np
##import pandas as pd
##
##data = np.array([[1, 2], [3, 4], [5, 6]])
##
##scaler = StandardScaler()
##scaled_data = scaler.fit_transform(data)
##print("Standardized data:\n", scaled_data)
##print("Mean:", scaler.mean_, "Std:", scaler.scale_)
##
##from sklearn.preprocessing import MinMaxScaler
##
##scaler = MinMaxScaler(feature_range=(0, 1))
##minmax_data = scaler.fit_transform(data)
##print("\nMin-Max scaled data:\n", minmax_data)
##print("Data min:", scaler.data_min_, "Data max:", scaler.data_max_)
##
##from sklearn.preprocessing import RobustScaler
##
##robust_scaler = RobustScaler()
##robust_data = robust_scaler.fit_transform(data)
##print("\nRobust scaled data:\n", robust_data)
##print("Median:", robust_scaler.center_, "IQR:", robust_scaler.scale_)
##
##data = np.array([1, 10, 100, 1000])
##
### Adding 1 to avoid log(0)
##log_data = np.log1p(data)
##print("\nLog transformed data:\n", log_data)
##
##from sklearn.preprocessing import KBinsDiscretizer
##
##est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
##binned_data = est.fit_transform(data.reshape(-1, 1))
##print("\nEqual-width bins:\n", binned_data.flatten())
##
##est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
##quantile_data = est.fit_transform(data.reshape(-1, 1))
##print("\nQuantile bins:\n", quantile_data.flatten())
##
##def clip_outliers(series, lower=0.05, upper=0.95):
##    lower_bound = series.quantile(lower)
##    upper_bound = series.quantile(upper)
##    return series.clip(lower_bound, upper_bound)
##
##data = pd.Series([1, 2, 3, 4, 5, 100])
##clipped_data = clip_outliers(data)
##print("\nClipped data:\n", clipped_data)
##
##data = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])
##
##robust_scaler = RobustScaler()
##robust_data = robust_scaler.fit_transform(data)
##print("\nRobust scaled data with outliers:\n", robust_data)
##
##from sklearn.preprocessing import PolynomialFeatures
##
##poly = PolynomialFeatures(degree=2, include_bias=False)
##poly_data = poly.fit_transform(data)
##print("\nPolynomial features:\n", poly_data)
##
##from sklearn.preprocessing import PolynomialFeatures
##
##interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
##interaction_data = interaction.fit_transform(data)
##print("\nInteraction terms:\n", interaction_data)

#--------------------------------------------------------
# data intergration, e.g., ads + user behavior + labels
#--------------------------------------------------------

##import pandas as pd
##import numpy as np
##from datetime import datetime, timedelta
##import random
##
### Set random seed for reproducibility
##np.random.seed(42)
##random.seed(42)
##
### ======================
### 1. Generate Realistic Datasets
### ======================
##TTL_SAMPLE_NUM=100
##RANGE_NUM0 = 500
##
### Generate TTL_SAMPLE_NUM users with demographics
##users = pd.DataFrame({
##    'user_id': [f'U{TTL_SAMPLE_NUM + i}' for i in range(TTL_SAMPLE_NUM)],
##    'age': np.random.randint(18, 70, TTL_SAMPLE_NUM),
##    'gender': np.random.choice(['M', 'F', 'Other'], TTL_SAMPLE_NUM, p=[0.48, 0.48, 0.04]),
##    'signup_date': [datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 400)) for _ in range(TTL_SAMPLE_NUM)],
##    'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], TTL_SAMPLE_NUM, p=[0.5, 0.2, 0.1, 0.1, 0.1])
##})
##
### Generate 50,000 activity events
##activities = pd.DataFrame({
##    'event_id': [f'E{1000 + i}' for i in range(RANGE_NUM0)],
##    'user_id': np.random.choice(users['user_id'], RANGE_NUM0),
##    'event_time': [datetime(2023, 1, 1) + timedelta(minutes=np.random.randint(0, 60 * 24 * 180)) for _ in range(RANGE_NUM0)],
##    'event_type': np.random.choice(['page_view', 'product_view', 'search', 'add_to_cart'], RANGE_NUM0,
##                                   p=[0.5, 0.3, 0.15, 0.05]),
##    'duration_sec': np.random.exponential(30, RANGE_NUM0).astype(int),
##    'product_id': [f'P{np.random.randint(1, 50)}' for _ in range(RANGE_NUM0)]
##})
##
### Generate 5,000 purchases
##purchases = pd.DataFrame({
##    'transaction_id': [f'T{RANGE_NUM0 + i}' for i in range(5000)],
##    'user_id': np.random.choice(users['user_id'], 5000),
##    'purchase_time': [datetime(2023, 1, 1) + timedelta(minutes=np.random.randint(0, 60 * 24 * 180)) for _ in
##                      range(5000)],
##    'product_id': [f'P{np.random.randint(1, 50)}' for _ in range(5000)],
##    'amount': np.round(np.random.lognormal(3, 0.5, 5000), 2),
##    'payment_method': np.random.choice(['credit_card', 'paypal', 'apple_pay'], 5000)
##})
##
##
### ======================
### 2. Data Integration Pipeline
### ======================
##
##def integrate_datasets(users, activities, purchases):
##    """Integrate user demographics, activity logs, and purchase records"""
##
##    # Convert all timestamps
##    for df in [users, activities, purchases]:
##        if 'time' in df.columns.str.contains('time|date'):
##            df[df.columns[df.columns.str.contains('time|date')]] = \
##                pd.to_datetime(df[df.columns[df.columns.str.contains('time|date')]].squeeze())
##
##    # 1. Calculate user activity features
##    activity_features = activities.groupby('user_id').agg(
##        total_events=('event_id', 'count'),
##        avg_duration=('duration_sec', 'mean'),
##        last_activity=('event_time', 'max'),
##        product_views=('event_type', lambda x: (x == 'product_view').sum()),
##        searches=('event_type', lambda x: (x == 'search').sum())
##    ).reset_index()
##
##    # 2. Calculate purchase behavior features
##    purchase_features = purchases.groupby('user_id').agg(
##        total_spend=('amount', 'sum'),
##        avg_purchase=('amount', 'mean'),
##        first_purchase=('purchase_time', 'min'),
##        last_purchase=('purchase_time', 'max'),
##        purchase_count=('transaction_id', 'count')
##    ).reset_index()
##
##    # 3. Calculate time since last activity/purchase
##    current_date = datetime(2023, 7, 1)
##    activity_features['days_since_activity'] = (current_date - activity_features['last_activity']).dt.days
##    purchase_features['days_since_purchase'] = (current_date - purchase_features['last_purchase']).dt.days
##
##    # 4. Merge all datasets
##    merged = users.merge(activity_features, on='user_id', how='left') \
##        .merge(purchase_features, on='user_id', how='left')
##
##    # 5. Fill NA values for users without activities/purchases
##    activity_cols = ['total_events', 'avg_duration', 'product_views', 'searches', 'days_since_activity']
##    purchase_cols = ['total_spend', 'avg_purchase', 'purchase_count', 'days_since_purchase']
##
##    merged[activity_cols] = merged[activity_cols].fillna(0)
##    merged[purchase_cols] = merged[purchase_cols].fillna(0)
##    merged[['first_purchase', 'last_purchase']] = merged[['first_purchase', 'last_purchase']].fillna(pd.NaT)
##
##    # 6. Calculate derived features
##    merged['purchase_probability'] = np.where(merged['purchase_count'] > 0, 1, 0)
##    merged['avg_duration'] = merged['avg_duration'].round(1)
##    merged['avg_purchase'] = merged['avg_purchase'].round(2)
##    merged['activity_level'] = pd.cut(merged['total_events'],
##                                      bins=[-1, 0, 5, 20, 100, np.inf],
##                                      labels=['none', 'low', 'medium', 'high', 'very_high'])
##
##    return merged
##
##
### ======================
### 3. Execute Integration
### ======================
##
### Run the integration pipeline
##final_dataset = integrate_datasets(users, activities, purchases)
##
### Show results
##print("Integrated dataset shape:", final_dataset.shape)
##print("\nSample records:")
##print(final_dataset.head(3))
##
##print("\nData types:")
##print(final_dataset.dtypes)
##
##print("\nMissing values:")
##print(final_dataset.isnull().sum())
##



#-----------------------------------------------------------------------
# train model
#-----------------------------------------------------------------------

def trainAndValidModel(name, model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds)

    return model, rmse

def testModel(name, model, evaluate_function, X_test, y_test):
    preds = model.predict(X_test)
    #rmse = mean_squared_error(y_valid, preds)
    metrics = evaluate_function(y_test, preds)

    return metrics

#-----------------------------------------------------------------------
# draw figure
#-----------------------------------------------------------------------
def getHeatMapv0(df, df_name, is_numeric_only=True, need_plot=False):
    print("{0}".format(df_name))
    corr = df.corr(numeric_only=is_numeric_only)
    getDatasetInfo(corr, "corr")
    if need_plot == True:
        sns.heatmap(corr, annot=True, fmt=".2f", square=True, linewidths=.5)
        plt.show()

def drawBoxplot(df, col_name, fig_size_a=8, fig_size_b=6, need_plot=False):
    plt.figure(figsize=(fig_size_a, fig_size_a))
    sns.boxplot(x=df[col_name])
    plt.title('Boxplot for {0}'.format(col_name))
    if need_plot == True:
        plt.show()

def drawScatterplot(df, col_name_x, col_name_y, fig_size_a=8, fig_size_b=6, need_plot=False):
    plt.figure(figsize=(fig_size_a, fig_size_a))
    sns.regplot(x=df[col_name_x], y=df[col_name_y], scatter_kws={'alpha': 0.5}, line_kws={"color": "darkblue"})
    plt.title('Scatter plot - {0} vs {1} with regression line'.format(col_name_x, col_name_y))
    plt.xlabel('{0}'.format(col_name_x))
    plt.ylabel('{0}'.format(col_name_y))

    if need_plot == True:
        plt.show()

def drawHistogram(df, col_name, fig_size_a=8, fig_size_b=6, this_color='g', num_of_bins=100, need_plot=False):
    plt.figure(figsize=(fig_size_a, fig_size_a))
    sns.hisplot(df[col_name], color=this_color, kde=True, bin=num_of_bins)
    plt.title('Histogram for {0}'.format(col_name))

    if need_plot == True:
        plt.show()

def drawXYFig(df, x_col, y_col, title="fig"):
    df.plt(x=x_col, y=y_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.show()

def drawSnsCountFig(df, x_col):
    sns.countplot(df, data=x_col)

def drawSnsDistFig(df, x_col):
    sns.distplot(df[x_col])

def drawSnsBoxFig(df, x_col):
    sns.boxplot(df[x_col])


#-----------------------------------------------------
# datetime extract day, hour, weekdays, etc
#-----------------------------------------------------

##import pandas as pd
##import numpy as np
##
### Create sample datetime data
##dates = pd.Series(pd.date_range('2023-01-01', periods=10, freq='12H'))  # 12-hour intervals
##
### Create DataFrame with extracted features
##dt_features = pd.DataFrame({
##    'original_datetime': dates,
##
##    # Date components
##    'year': dates.dt.year,
##    'month': dates.dt.month,
##    'day': dates.dt.day,
##
##    # Time components
##    'hour': dates.dt.hour,
##    'minute': dates.dt.minute,
##    'second': dates.dt.second,
##
##    # Week information
##    'weekday': dates.dt.weekday,  # Monday=0, Sunday=6
##    'weekday_name': dates.dt.day_name(),  # Full day name
##    'is_weekend': dates.dt.weekday >= 5,  # Saturday/Sunday
##
##    # Week of year
##    'week_of_year': dates.dt.isocalendar().week,
##
##    # Quarter
##    'quarter': dates.dt.quarter,
##
##    # Special day indicators
##    'is_month_start': dates.dt.is_month_start,
##    'is_month_end': dates.dt.is_month_end,
##
##    # Time of day categories
##    'time_of_day': pd.cut(dates.dt.hour,
##                          bins=[0, 6, 12, 18, 24],
##                          labels=['Night', 'Morning', 'Afternoon', 'Evening'],
##                          right=False)
##})
##
##print("dt_features")
##print(dt_features)
##print('\n')

#-----------------------------------------------------------------------
# text processing
#-----------------------------------------------------------------------

def removePuncFromWord(word):
    clean_word_list = []
    for chr in word:
        if chr.isalpha() == False:
            continue
        clean_word_list.append(chr)
    clean_word = "".join(clean_word_list)
    return clean_word

def removePunctuation(reviews):
    lst = reviews.split()
    new_lst = []

    for word in lst:
        if word not in string.punctuation:
            word = removePuncFromWord(word)
            new_lst.append(word)

    clean_review = ' '.join(new_lst)
    return  clean_review

def turnLower(review):
    new_lst = []
    review_lst = review.split()
    for word in review_lst:
        word = word.lower()
        new_lst.append(word)
    new_review = ' '.join(new_lst)
    return new_review

def tokenize(reviews):
    tokens = reviews.split()
    return tokens

#stopwords = nltk.corpus.stopwords.words('english')


#import spacy
## Load the English language model (download first if needed: `python -m spacy download en_core_web_sm`)
#nlp = spacy.load("en_core_web_sm")
#stopwords = nlp.Defaults.stop_words  # Returns a set

'''
stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}
'''

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords = ENGLISH_STOP_WORDS  # Returns a frozen set

def removeStopwords(str0):
    list0 = tokenize(str0)
    word_list_final = []
    for x in list0:
        if x not in stopwords:
            word_list_final.append(x)

    return ' '.join(word_list_final)

def Lemmatize(tokens):
    lemmatized_token_list = []
    for token in tokens:
        lemmatized_token = wordnet_lemmatizer.lemmatize(token)
        lemmatized_token_list.append(lemmatized_token)
    return lemmatized_token_list

def LemmatizeStr(str0):
    list0 = tokenize(str0)
    list1 = []
    for token in list0:
        lemmatized_token = wordnet_lemmatizer.lemmatize(token)
        list1.append(lemmatized_token)

    return ' '.join(list1)


#------------------------------------------------
# text data processing
#------------------------------------------------
TRAINING_DATASET_SIZE=1000

def loadDataset(path):
    corpus = []
    count = 0
    with open(path, encoding='utf-8', erros='ignore') as f:
        for i in f:
            if count < TRAINING_DATASET_SIZE:
                corpus.append(i)
                count += 1

        return corpus

def cleanAndTokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

def tokenize(sentence):
    rst = re.findall(r'\b\w+\b', sentence.lower())
    return rst


def buildVocab(texts, min_freq=1, special_tokens=['<pad>', '<unk>']):
    """
    Build vocabulary from text data

    Args:
        texts: List of strings or list of token lists
        min_freq: Minimum frequency for a token to be included
        max_size: Maximum vocabulary size (None for no limit)
        special_tokens: List of special tokens to add first

    Returns:
        vocab: Dictionary {token: index}
        reverse_vocab: Dictionary {index: token}
    """
    # Initialize vocabulary with special tokens
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    token_counts = Counter()

    # Process texts
    for text in texts:
        # If input is string, tokenize it
        if isinstance(text, str):
            tokens = tokenize(text)  # Basic tokenization
        else:
            tokens = text

        token_counts.update(tokens)

    # Add tokens meeting frequency threshold
    for token, count in token_counts.items():
        if count >= min_freq:
            if token not in vocab:
                vocab[token] = len(vocab)

    # Create reverse mapping
    reverse_vocab = {idx: token for token, idx in vocab.items()}

    return vocab, reverse_vocab

def encodeSentence(sentence, vocab):
    tokens = tokenize(sentence)
    rst = [vocab.get(token, vocab['<unk>']) for token in tokens]

def padSentences(encoded_sentences, max_len=100):
    rst = []
    for seq in encoded_sentences:
        tmp = None
        if len(seq) < max_len:
            tmp = seq + [0] * (max_len - len(seq))
        else:
            tmp = seq[:max_len]
        rst.append(tmp)
    return rst

def sentencesToTensor(sentences, vocab, max_len=None):
    encoded = []
    for seq in sentences:
        encoded.append(encodeSentence(seq, vocab))

    if not max_len:
        max_len = max(len(seq) for seq in sentences)

    padded = padSentences(encoded, max_len)
    return torch.tensor(padded, dtype=torch.long)


#----------------------------------------
# text embedding
#----------------------------------------
##import torch
##import torch.nn as nn
##import torch.optim as optim
##from collections import Counter
##import numpy as np
##
### Sample corpus
##corpus = [
##    "word embeddings are cool",
##    "word2vec is a popular embedding model",
##    "pytorch makes deep learning easy"
##]
##
### Hyperparameters
##EMBEDDING_DIM = 100
##WINDOW_SIZE = 2
##BATCH_SIZE = 32
##EPOCHS = 100
##
##
### Preprocessing
##def preprocess(corpus):
##    words = []
##    for sentence in corpus:
##        words.extend(sentence.lower().split())
##    vocab = Counter(words)
##    vocab = sorted(vocab, key=vocab.get, reverse=True)
##    word2idx = {word: i for i, word in enumerate(vocab)}
##    idx2word = {i: word for i, word in enumerate(vocab)}
##    return word2idx, idx2word, vocab
##
##
##word2idx, idx2word, vocab = preprocess(corpus)
##VOCAB_SIZE = len(vocab)
##
##
### Generate training pairs
##def create_training_data(corpus, word2idx, window_size):
##    training_data = []
##    for sentence in corpus:
##        sentence = sentence.lower().split()
##        for i, target_word in enumerate(sentence):
##            for j in range(i - window_size, i + window_size + 1):
##                if j != i and 0 <= j < len(sentence):
##                    context_word = sentence[j]
##                    training_data.append((word2idx[target_word], word2idx[context_word]))
##    return training_data
##
##
##training_data = create_training_data(corpus, word2idx, WINDOW_SIZE)
##
##
### Word2Vec model
##class Word2Vec(nn.Module):
##    def __init__(self, vocab_size, embedding_dim):
##        super(Word2Vec, self).__init__()
##        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
##        self.linear = nn.Linear(embedding_dim, vocab_size)
##
##    def forward(self, x):
##        x = self.embeddings(x)
##        x = self.linear(x)
##        return x
##
##
##model = Word2Vec(VOCAB_SIZE, EMBEDDING_DIM)
##criterion = nn.CrossEntropyLoss()
##optimizer = optim.Adam(model.parameters(), lr=0.001)
##
### Training loop
##for epoch in range(EPOCHS):
##    total_loss = 0
##    for target, context in training_data:
##        target_tensor = torch.tensor([target], dtype=torch.long)
##        context_tensor = torch.tensor([context], dtype=torch.long)
##
##        optimizer.zero_grad()
##        output = model(target_tensor)
##        loss = criterion(output, context_tensor)
##        loss.backward()
##        optimizer.step()
##
##        total_loss += loss.item()
##
##    if (epoch + 1) % 10 == 0:
##        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(training_data):.4f}')
##
### Get word embeddings
##embeddings = model.embeddings.weight.data
##print(f"Embedding for 'word': {embeddings[word2idx['word']][:10]}")  # First 10 dimensions
##
##import gensim.downloader as api
##from torch.utils.data import Dataset, DataLoader
##
### Load pre-trained model
##w2v_model = api.load('word2vec-google-news-300')
##
##
### Create PyTorch embedding layer
##class PretrainedEmbeddingLayer(nn.Module):
##    def __init__(self, word2vec_model, freeze=True):
##        super().__init__()
##        vocab_size = len(word2vec_model.key_to_index)
##        embedding_dim = word2vec_model.vector_size
##
##        # Create embedding matrix
##        embedding_matrix = torch.zeros((vocab_size, embedding_dim))
##        for i, word in enumerate(word2vec_model.key_to_index):
##            embedding_matrix[i] = torch.tensor(word2vec_model[word])
##
##        self.embedding = nn.Embedding.from_pretrained(
##            embedding_matrix,
##            freeze=freeze,
##            padding_idx=0
##        )
##
##    def forward(self, x):
##        return self.embedding(x)
##
##
### Example usage
##embedding_layer = PretrainedEmbeddingLayer(w2v_model)
##print(f"Embedding layer shape: {embedding_layer.embedding.weight.shape}")
##
##
##from transformers import BertModel, BertTokenizer
##import torch
##
### Initialize tokenizer and model
##tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
##model = BertModel.from_pretrained('bert-base-uncased')
##
### Input text
##text = "Natural language processing with PyTorch is powerful"
##
### Tokenize and prepare input
##inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
##
### Get embeddings
##with torch.no_grad():
##    outputs = model(**inputs)
##
### Last hidden state (sequence embeddings)
##last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
##print(f"BERT embedding shape: {last_hidden_state.shape}")
##
### Pooled output (sentence embedding)
##pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]
##print(f"Pooled output shape: {pooled_output.shape}")
##
### Get specific token embedding
##token_embeddings = last_hidden_state[0]  # First sequence
##print(f"Embedding for 'language': {token_embeddings[2][:10]}")  # First 10 dimensions

#-----------------------
#token embedding
#-----------------------

##import torch
##from collections import defaultdict
##
### Sample text
##text = "PyTorch makes deep learning easy and fun"
##
### Create vocabulary
##word_to_idx = defaultdict(lambda: len(word_to_idx))
##word_to_idx['<unk>'] = 0  # Unknown token
##word_to_idx['<pad>'] = 1  # Padding token
##
### Tokenize
##tokens = text.lower().split()
##indices = [word_to_idx[word] for word in tokens]
##
##print("Vocabulary:", dict(word_to_idx))
##print("Token indices:", indices)
##
### Convert to PyTorch tensor
##token_tensor = torch.tensor(indices)
##print("PyTorch tensor:", token_tensor)

#-----------------------
#using glove
#-----------------------
# Load Glove
glove_path = ""
#glove_enbeddings = loadGloveEmbeddings(glove_path)

def loadGloveEmbeddings(file_path = glove_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def tokensToEmbeddings(tokens, glove, embedding_dim=128):
    vectors = []
    for token in tokens:
        if token in glove:
            vectors.append(glove[token])
        else:
            # unknown word -> zero vector or random
            vectors.append(np.zeros(embedding_dim))
    return np.array(vectors)

#--------------------------------------------------------
# feature engineering
#--------------------------------------------------------

#-------------------------------
# interaction features
#-------------------------------

##import pandas as pd
##import numpy as np
##
### Sample data
##df = pd.DataFrame({
##    'age': [25, 30, 35, 40],
##    'income': [50000, 80000, 60000, 90000],
##    'credit_score': [650, 700, 720, 680]
##})
##
### Create interaction terms
##df['age_income_interaction'] = df['age'] * df['income']
##df['age_credit_interaction'] = df['age'] * df['credit_score']
##print("Interaction Features:")
##print(df[['age', 'income', 'age_income_interaction']].head())
##
### Create crossed feature by combining categories
##df['location'] = ['NY', 'CA', 'NY', 'TX']
##df['gender'] = ['M', 'F', 'F', 'M']
##df['location_gender_cross'] = df['location'] + '_' + df['gender']
##print("\nFeature Cross Example:")
##print(df[['location', 'gender', 'location_gender_cross']].head())
##
### Create ratio features
##df['income_to_age_ratio'] = df['income'] / df['age']
##df['credit_to_income_ratio'] = df['credit_score'] / df['income'] * 1000
##print("\nRatio Features:")
##print(df[['income', 'age', 'income_to_age_ratio']].head())
##
### Create polynomial features manually
##df['age_squared'] = df['age'] ** 2
##df['income_squared'] = df['income'] ** 2
##print("\nQuadratic Features:")
##print(df[['age', 'age_squared', 'income', 'income_squared']].head())
##
##from sklearn.preprocessing import PolynomialFeatures
##
### Create polynomial features (degree=2 includes squares and interactions)
##poly = PolynomialFeatures(degree=2, include_bias=False)
##poly_features = poly.fit_transform(df[['age', 'income']])
##poly_df = pd.DataFrame(poly_features,
##                      columns=['age', 'income', 'age^2', 'age*income', 'income^2'])
##print("\nscikit-learn Polynomial Features:")
##print(poly_df.head())
##
### Financial domain example
##df['debt_to_income'] = df['income'] * 0.3 - df['credit_score']  # Hypothetical formula
##print("\nDomain-Specific Features:")
##print(df[['income', 'credit_score', 'debt_to_income']].head())
##
### For time series data
##df['date'] = pd.date_range('2023-01-01', periods=4)
##df['days_since'] = (df['date'].max() - df['date']).dt.days
##df['income_per_day'] = df['income'] / df['days_since']
##print("\nTime-Based Ratios:")
##print(df[['date', 'income', 'income_per_day']])
##
##print("df")
##print(df)
##print("\n")

#-------------------------------
# statistical aggregation
#-------------------------------

##import pandas as pd
##import numpy as np
##
### Sample DataFrame
##df = pd.DataFrame({
##    'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
##    'Value1': [10, 15, 20, 10, 25, 30, 10, 20],
##    'Value2': [100, 150, 200, 100, 250, 300, 100, 200]
##})
##
### Group by 'Category' and calculate multiple stats
##grouped = df.groupby('Category').agg({
##    'Value1': ['mean', 'max', 'count'],
##    'Value2': ['min', 'std', 'sum']
##})
##
##print("Pandas GroupBy with Multiple Aggregations:")
##print(grouped)
##
### Group by 'Category' and calculate multiple stats
##grouped1 = df.groupby('Category').agg(
##    ['mean', 'max', 'count', 'min', 'std', 'sum']
##)
##
##print("Pandas GroupBy with Multiple Aggregations:")
##print(grouped1)
##
### More readable syntax with named aggregations
##result = df.groupby('Category').agg(
##    mean_value1=('Value1', 'mean'),
##    max_value1=('Value1', 'max'),
##    count_value1=('Value1', 'count'),
##    range_value2=('Value2', lambda x: x.max() - x.min())
##)
##
##print("\nNamed Aggregations:")
##print(result)
##
### Add another grouping column
##df['Group'] = ['X', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y']
##
### Group by multiple columns
##multi_group = df.groupby(['Category', 'Group']).agg({
##    'Value1': ['mean', 'max', 'count'],
##    'Value2': 'sum'
##})
##
##print("\nMultiple Grouping Columns:")
##print(multi_group)

#---------------------------------
# sequence feature
#---------------------------------

##import pandas as pd
##import numpy as np
##from collections import defaultdict
##
### Sample user browsing history
##data = {
##    'user_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
##    'timestamp': [
##        '2023-01-01 10:00:00', '2023-01-01 10:01:00', '2023-01-01 10:03:00',
##        '2023-01-01 10:05:00', '2023-01-01 10:06:00', '2023-01-01 11:00:00',
##        '2023-01-01 11:01:30', '2023-01-01 11:02:00', '2023-01-01 11:04:00',
##        '2023-01-01 12:00:00', '2023-01-01 12:02:00', '2023-01-01 12:05:00'
##    ],
##    'page_url': [
##        'home', 'products', 'product_A', 'cart', 'checkout',
##        'home', 'categories', 'product_B', 'cart',
##        'home', 'product_C', 'categories'
##    ],
##    'dwell_time': [30, 45, 120, 60, 90, 40, 50, 180, 75, 60, 150, 80]
##}
##
##df = pd.DataFrame(data)
##df['timestamp'] = pd.to_datetime(df['timestamp'])
##
##def generate_sliding_windows(events, window_size=3, stride=1):
##    """Generate sliding windows from a sequence of events"""
##    windows = []
##    for i in range(0, len(events) - window_size + 1, stride):
##        window = events[i:i + window_size]
##        windows.append(window)
##    return windows
##
### Group by user and generate windows
##user_windows = defaultdict(list)
##for user_id, group in df.groupby('user_id'):
##    events = group['page_url'].tolist()
##    windows = generate_sliding_windows(events, window_size=3, stride=1)
##    user_windows[user_id] = windows
##
##print("User click sequence windows:")
##for user, windows in user_windows.items():
##    print(f"User {user}:")
##    for i, window in enumerate(windows):
##        print(f"  Window {i+1}: {window}")
##
##
##def generate_time_windows(user_df, time_window='5min'):
##    """Generate time-based sliding windows"""
##    user_df = user_df.sort_values('timestamp')
##    windows = []
##
##    # Create rolling time window
##    for i in range(len(user_df)):
##        window_mask = (user_df['timestamp'] >= user_df['timestamp'].iloc[i]) & \
##                      (user_df['timestamp'] <= user_df['timestamp'].iloc[i] + pd.Timedelta(time_window))
##        window = user_df[window_mask]
##        if len(window) > 1:  # Only consider windows with at least 2 events
##            windows.append(window)
##
##    return windows
##
##
### Apply to each user
##time_windows = defaultdict(list)
##for user_id, group in df.groupby('user_id'):
##    windows = generate_time_windows(group, time_window='5min')
##    time_windows[user_id] = windows
##
##print("\nTime-based windows (first user):")
##for i, window in enumerate(time_windows[1][:3]):  # Show first 3 windows for user 1
##    print(f"Window {i + 1}:")
##    print(window[['timestamp', 'page_url', 'dwell_time']])
##
##
##def extract_sequence_features(windows):
##    """Extract features from click sequences"""
##    features = []
##    for window in windows:
##        # Basic sequence features
##        seq_length = len(window)
##        unique_pages = len(set(window))
##
##        # Transition features
##        transitions = [f"{window[i]}=>{window[i + 1]}" for i in range(len(window) - 1)]
##
##        # Dwell time features (if available)
##        dwell_times = [dwell for dwell in window['dwell_time']] if isinstance(window, pd.DataFrame) else [0] * len(
##            window)
##
##        features.append({
##            'sequence': ' → '.join(window) if not isinstance(window, pd.DataFrame) else ' → '.join(window['page_url']),
##            'length': seq_length,
##            'unique_pages': unique_pages,
##            'transitions': transitions,
##            'avg_dwell_time': np.mean(dwell_times),
##            'total_dwell_time': np.sum(dwell_times)
##        })
##    return features
##
##
### Extract features for all users
##all_features = {}
##for user_id, windows in user_windows.items():
##    all_features[user_id] = extract_sequence_features(windows)
##
##print("\nExtracted features for user 1:")
##print(pd.DataFrame(all_features[1]).head())
##
##
##def build_transition_matrix(df):
##    """Build a Markov transition matrix for page transitions"""
##    # Create all possible transitions
##    df['next_page'] = df.groupby('user_id')['page_url'].shift(-1)
##    transitions = df.dropna(subset=['next_page'])
##
##    # Create transition matrix
##    all_pages = pd.unique(df['page_url'])
##    trans_matrix = pd.DataFrame(0, index=all_pages, columns=all_pages)
##
##    for _, row in transitions.iterrows():
##        trans_matrix.loc[row['page_url'], row['next_page']] += 1
##
##    # Normalize to probabilities
##    trans_matrix = trans_matrix.div(trans_matrix.sum(axis=1), axis=0).fillna(0)
##    return trans_matrix
##
##
##transition_matrix = build_transition_matrix(df)
##print("\nTransition Probability Matrix:")
##print(transition_matrix)
##
##
##def detect_sessions(df, inactivity_threshold='30min'):
##    """Identify browsing sessions based on inactivity"""
##    df = df.sort_values(['user_id', 'timestamp'])
##
##    # Calculate time difference between consecutive events
##    df['time_diff'] = df.groupby('user_id')['timestamp'].diff()
##
##    # New session starts when time difference > threshold or new user
##    df['new_session'] = (df['time_diff'] > pd.Timedelta(inactivity_threshold)) | (df['user_id'].diff() != 0)
##    df['session_id'] = df['new_session'].cumsum()
##
##    return df
##
##
##session_df = detect_sessions(df)
##print("\nSession detection results:")
##print(session_df[['user_id', 'timestamp', 'page_url', 'session_id']])
##
##from mlxtend.preprocessing import TransactionEncoder
##from mlxtend.frequent_patterns import apriori
##
### Prepare transaction data for pattern mining
##transactions = []
##for user, windows in user_windows.items():
##    for window in windows:
##        transactions.append(window)
##
### Convert to one-hot encoded format
##te = TransactionEncoder()
##te_ary = te.fit(transactions).transform(transactions)
##df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
##
### Find frequent patterns
##frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)
##print("\nFrequent click patterns:")
##print(frequent_itemsets.sort_values('support', ascending=False))
##
##import networkx as nx
##import matplotlib.pyplot as plt
##
### Create graph from transition matrix
##G = nx.DiGraph()
##
##for source in transition_matrix.index:
##    for target in transition_matrix.columns:
##        weight = transition_matrix.loc[source, target]
##        if weight > 0:
##            G.add_edge(source, target, weight=weight)
##
### Draw the graph
##plt.figure(figsize=(10, 8))
##pos = nx.spring_layout(G)
##nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
##nx.draw_networkx_edges(G, pos, edge_color='gray', width=[d['weight']*10 for (u,v,d) in G.edges(data=True)])
##nx.draw_networkx_labels(G, pos, font_size=12)
##edge_labels = {(u,v): f"{d['weight']:.2f}" for (u,v,d) in G.edges(data=True)}
##nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
##plt.title("User Click Transition Probabilities")
##plt.axis('off')
##plt.show()

#--------------------------------------------------------
#
#--------------------------------------------------------

#--------------------------------------------------------
#Model
#--------------------------------------------------------

#----------------------------
# simple transformer for classification
#----------------------------

#hyper parameters
seq_len = 10
vocab_size = 100
embed_dim = 32
num_heads = 4
ff_dim = 64
num_layers = 2
batch_size = 8
num_classes = 2
lr = 1e-3
epochs = 10


#---------- positional embedding -------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

#--------------------------------
# Simple Transformer for classification
#--------------------------------

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, src):
        # src shape: (seq_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Take the mean across sequence dimension
        output = output.mean(dim=0)
        return self.fc(output)

# --- train function ---
def trainTransformerClassifierV0(model, X, y):
    model.train()
    outputs = model(X)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training step
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()

    return loss.item()

def predictTransformerClassifierV0(model, X):
    model.eval()
    with torch.no_grad:
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
    return preds

# Example usage
#if __name__ == "__main__":
#    # Hyperparameters
#    vocab_size = 10000  # Size of your vocabulary
#    batch_size = 32
#    seq_len = 50
#    num_classes = 2  # Binary classification
#
#    # Create model
#    model = TransformerClassifier(
#        vocab_size=vocab_size,
#        d_model=128,
#        nhead=8,
#        num_layers=3,
#        num_classes=num_classes
#    )
#
#    optimizer = optim.Adam(model.parameters(), lr=lr)
#    criterion = nn.CrossEntropyLoss()
#
#    # Create dummy data (replace with real data)
#    # Shape: (seq_len, batch_size)
#    x = torch.randint(0, vocab_size, (seq_len, batch_size))
#    # Dummy labels
#    y = torch.randint(0, num_classes, (batch_size,))
#
#    # Forward pass
#    outputs = model(x)
#    print(f"Output shape: {outputs.shape}")  # Should be [batch_size, #num_classes]
#
#    # Training setup example
#    for epoch in range(epochs):
#        loss = trainTransformerClassifierV0(model, x, y)
#        if epoch % 2 == 0:
#            print(f"Epoch {epoch}: loss = {loss:.4f}")
#
#    preds = predictTransformerClassifierV0(model, x)
#    print("predictions:", preds.tolist())
#    print("ground truth:", y.tolist())

# --- Instantiate model ---
# model = TransformerClassifier(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=3, num_classes=num_classes)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()

# Dummy dataset (classification task)
#X = torch.randint(0, vocab_size, (batch_size, seq_len)) # token_indices
#y = torch.randint(0, num_classes, (batch_size,))

# --- Run training ---
#for epoch in range(epochs):
#    loss = train(model, X, y)
#    if epoch % 2 == 0:
#        print("Epoch {epoch}: loss = {loss:.4f}")

# --- Predict ---
#preds = predict(model, X)
#print("Predictions:", preds.tolist())
#print("Ground truth:", y.tolist())


#--------------------------------
# Simple Transformer for classification
#--------------------------------

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.linear_out = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.linear_in(src) * math.sqrt(self.d_model)
        output = self.transformer(src)
        #output = output.mean(dim=1)  # Average over sequence
        return self.linear_out(output)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def trainTransformerRegressionV0(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def predictTransformerRegressionV0(model, dataloader, device='cpu'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            output = model(X)
            predictions.append(output.cpu())
    return torch.cat(predictions)


# Example usage
#if __name__ == "__main__":
#    # 1. Generate synthetic time series data
#    seq_len = 20
#    n_samples = 1000
#    input_dim = 5  # Number of features per timestep
#
#    # Random data (replace with your actual data)
#    X = np.random.randn(n_samples, seq_len, input_dim)
#    y = np.random.randn(n_samples)  # Regression target
#
#    # 2. Preprocess data
#    scaler = StandardScaler()
#    X = scaler.fit_transform(X.reshape(-1, input_dim)).reshape(n_samples, #seq_len, input_dim)
#    y = (y - y.mean()) / y.std()
#
#    # 3. Split data
#    train_size = int(0.8 * n_samples)
#    X_train, X_test = X[:train_size], X[train_size:]
#    y_train, y_test = y[:train_size], y[train_size:]
#
#    # 4. Create dataloaders
#    train_dataset = TimeSeriesDataset(X_train, y_train)
#    test_dataset = TimeSeriesDataset(X_test, y_test)
#    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#    test_loader = DataLoader(test_dataset, batch_size=32)
#
#    # 5. Initialize model
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    model = TransformerRegressor(input_dim=input_dim, d_model=64, #nhead=4, num_layers=2).to(device)
#    criterion = nn.MSELoss()
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#    # 6. Training loop
#    n_epochs = 10
#    for epoch in range(n_epochs):
#        train_loss = train(model, train_loader, criterion, optimizer, #device)
#        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}')
#
#    # 7. Evaluation
#    predictions = predict(model, test_loader, device)
#    test_loss = criterion(predictions, torch.FloatTensor(y_test))
#    print(f'Test MSE: {test_loss.item():.4f}')
#
#    # 8. Make new predictions
#    new_data = np.random.randn(3, seq_len, input_dim)  # 3 new samples
#    new_data = scaler.transform(new_data.reshape(-1, #input_dim)).reshape(3, seq_len, input_dim)
#    new_loader = DataLoader(TimeSeriesDataset(new_data, np.zeros(3)), #batch_size=3)
#    preds = predict(model, new_loader, device)
#    print('New predictions:', preds)

#--------------------------------
# change numerical features to embedding
#--------------------------------

# assuming there are 10 numerical features, target is to map them to a embedding with dim=8
class NumericEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.linear(x)

# data
#batch_size = 4
#num_features = 10
#x_numeric = torch.randn(batch_size, num_features)

#initialize embedding
#embedding_layer = NumericEmbedding(input_dim=10, embedding_dim=8)
#embedded = embedding_layer(x_numeric)
#print(embedded.shape) # torch.size([4, 8])


#--------------------------------
# change numerical features to bins, then to embedding
#--------------------------------
class BinedEmbedding(nn.Module):
    def __init__(self, num_bins_list, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_bins, embedding_dim) for num_bins in num_bins_list
        ])

    def forward(self, x_binned):
        # x_binned: LongTensor, shape (batch_size, num_features)
        rst = []
        for idx, embedding in enumerate(self.embeddings):
            tmp = embedding(x_binned[:, idx])
            rst.append(tmp)
        return torch.cat(rst, dim=-1)

#-------------
#case 1:
#-------------
# simulation data: 3 numerical features, after binned
#x_binned = torch.tensor([
# [1,4,2],
# [0,3,7],
# [2,5,1],
# ], dtype=torch.long)
# )

# the bin number of each feature is 10, the output embedding dim is 4
#model = BinnedEmbedding(num_bins_list=[10, 10, 10], embedding_dim=4)
#output = model(x)
#print(output.shape) % output: torch.Size([3, 12]), 3 features, 4 dim size each

#-------------
#case 2:
#-------------
# simulation data: 3 numerical features, after binned
#x_binned = torch.tensor([
# [1,4,2],
# [0,3,1],
# [2,5,0],
# [1,1,1],
# [0,2,2],
# ], dtype=torch.long) # shape(5,3)
# )
#
#y = np.array([0,1,0,1,0]) # classification target
#
#
# the number of categories in each column
#num_bins = [3,6,3]
#embedding_dim = 4
#
#create embedding layers (rach feature has one embedding)
#embeddings = nn.ModuleList([
#                 nn.Embedding(num_embeddings = n, embedding_dim=embedding_dim)
#                 for n in num_bins
#        ])
#
#with torch.no_grad():
#    embeddings = []
#    for idx, emb in enumerate(embeddings):
#        tmp = emd(x_binned[:, idx])
#        embeddings.append(tmp)
#    x_embed = torch.cat(embeddings,dim=1) # shape: (5, 3*4)
#    x_np = x_embed.numpy() # change to numpy, feed to xgboost
#
#X_train, X_test, y_train, y_test = train_test_split(x_np, y, test_size=0.4, random_state=42)
#
#model = xgb.XGBClassifier()
#model.fit(X_train, y_train)
#predict
#y_pred = model.predict(X_test)
#
#from sklearn.metrics import accuracy_score
#print("Accuracy:", accuracy_score(y_test, y_pred))


#--------------------------------------------------------
# embeddingAsInputToXgboostClassifierV0.py
#--------------------------------------------------------

##import numpy as np
##import torch
##import torch.nn as nn
##from torch.utils.data import DataLoader, TensorDataset
##from sklearn.model_selection import train_test_split
##from xgboost import XGBClassifier  # or XGBRegressor
##from sklearn.metrics import accuracy_score
##
### ============================================
### 1. Generate Synthetic Data
### ============================================
##num_samples = 1000
##num_features = 20
##num_classes = 3
##
### Create random data
##X = np.random.randn(num_samples, num_features)
##y = np.random.randint(0, num_classes, size=num_samples)
##
### Split into train/test
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
##
### Convert to PyTorch tensors
##X_train_tensor = torch.FloatTensor(X_train)
##y_train_tensor = torch.LongTensor(y_train)
##X_test_tensor = torch.FloatTensor(X_test)
##
##
### ============================================
### 2. Create Embedding Model (PyTorch)
### ============================================
##class EmbeddingModel(nn.Module):
##    def __init__(self, input_dim, embedding_dim):
##        super().__init__()
##        self.net = nn.Sequential(
##            nn.Linear(input_dim, 128),
##            nn.ReLU(),
##            nn.Linear(128, embedding_dim)
##        )
##
##    def forward(self, x):
##        return self.net(x)
##
##
### Initialize model
##embedding_dim = 16
##embedder = EmbeddingModel(num_features, embedding_dim)
##
##
### ============================================
### 3. Train Embedding Model (Optional)
### ============================================
### Only needed if you want task-specific embeddings
##def train_embedder(model, X, y, epochs=10):
##    model.train()
##    criterion = nn.CrossEntropyLoss()
##    optimizer = torch.optim.Adam(model.parameters())
##    dataset = TensorDataset(X, y)
##    loader = DataLoader(dataset, batch_size=32, shuffle=True)
##
##    for epoch in range(epochs):
##        for batch_X, batch_y in loader:
##            optimizer.zero_grad()
##            outputs = model(batch_X)
##            loss = criterion(outputs, batch_y)
##            loss.backward()
##            optimizer.step()
##        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
##
##
### Uncomment to train:
##train_embedder(embedder, X_train_tensor, y_train_tensor)
##
### ============================================
### 4. Generate Embeddings
### ============================================
##def get_embeddings(model, X):
##    model.eval()
##    with torch.no_grad():
##        return model(X).numpy()
##
##
##X_train_emb = get_embeddings(embedder, X_train_tensor)
##X_test_emb = get_embeddings(embedder, X_test_tensor)
##
### ============================================
### 5. Train XGBoost on Embeddings
### ============================================
##xgb_model = XGBClassifier(n_estimators=100, max_depth=3)
##xgb_model.fit(X_train_emb, y_train)
##
### Evaluate
##y_pred = xgb_model.predict(X_test_emb)
##accuracy = accuracy_score(y_test, y_pred)
##print(f"XGBoost Accuracy: {accuracy:.4f}")
##
### Feature importance
##print("\nFeature Importances:")
##for i, imp in enumerate(xgb_model.feature_importances_):
##    print(f"Embedding dim {i}: {imp:.4f}")##

#-------------------------------------------------------
# embeddingAsInputToXgboostRegressorV0.py
#-------------------------------------------------------
##import numpy as np
##import torch
##import torch.nn as nn
##from torch.utils.data import DataLoader, TensorDataset
##from sklearn.model_selection import train_test_split
##from xgboost import XGBRegressor
##from sklearn.metrics import mean_squared_error, r2_score
##
### ============================================
### 1. Generate Synthetic Regression Data
### ============================================
##num_samples = 1000
##num_features = 20
##
### Create random data with a simple relationship
##X = np.random.randn(num_samples, num_features)
### Create target with some non-linearity
##y = 5 * X[:, 0] + 2 * np.sin(X[:, 1]) + 0.5 * np.random.randn(##num_samples)
##
### Split into train/test
##X_train, X_test, y_train, y_test = train_test_split(X, y, ##test_size=0.2, random_state=42)
##
### Convert to PyTorch tensors
##X_train_tensor = torch.FloatTensor(X_train)
##y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # XGB needs 1D ##array
##X_test_tensor = torch.FloatTensor(X_test)
##
##
### ============================================
### 2. Create Embedding Model (PyTorch)
### ============================================
##class Autoencoder(nn.Module):
##    def __init__(self, input_dim, embedding_dim):
##        super().__init__()
##        self.encoder = nn.Sequential(
##            nn.Linear(input_dim, 128),
##            nn.ReLU(),
##            nn.Linear(128, embedding_dim)
##        )
##        self.decoder = nn.Sequential(
##            nn.Linear(embedding_dim, 128),
##            nn.ReLU(),
##            nn.Linear(128, input_dim)
##        )
##
##    def forward(self, x):
##        encoded = self.encoder(x)
##        decoded = self.decoder(encoded)
##        return decoded
##
##
### Initialize model
##embedding_dim = 10  # Reduced dimension
##
### Initialize as autoencoder
##embedder = Autoencoder(num_features, embedding_dim)
##
##
### ============================================
### 3. Train Embedding Model (Optional)
### ============================================
##def train_embedder(model, X, y, epochs=20):
##    model.train()
##    criterion = nn.MSELoss()
##    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
##    dataset = TensorDataset(X, y)
##    loader = DataLoader(dataset, batch_size=32, shuffle=True)
##
##    for epoch in range(epochs):
##        for batch_X, batch_y in loader:
##            optimizer.zero_grad()
##            outputs = model(batch_X)
##            loss = criterion(outputs, batch_y)
##            loss.backward()
##            optimizer.step()
##        if epoch % 5 == 0:
##            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
##
##
### Train as autoencoder (reconstruct original features)
##train_embedder(embedder, X_train_tensor, X_train_tensor)
##
##
### ============================================
### 4. Generate Embeddings
### ============================================
##def get_embeddings(model, X):
##    model.eval()
##    with torch.no_grad():
##        return model(X).numpy()
##
##
##X_train_emb = get_embeddings(embedder, X_train_tensor)
##X_test_emb = get_embeddings(embedder, X_test_tensor)
##
### ============================================
### 5. Train XGBoost Regressor on Embeddings
### ============================================
##xgb_model = XGBRegressor(
##    n_estimators=150,
##    max_depth=5,
##    learning_rate=0.1,
##    random_state=42
##)
##
##xgb_model.fit(X_train_emb, y_train)  # y_train is original targets
##
### Evaluate
##y_pred = xgb_model.predict(X_test_emb)
##mse = mean_squared_error(y_test, y_pred)
##r2 = r2_score(y_test, y_pred)
##
##print(f"\nXGBoost Regression Results:")
##print(f"MSE: {mse:.4f}")
##print(f"R² Score: {r2:.4f}")
##
### Feature importance
##print("\nEmbedding Dimension Importance:")
##for i, imp in enumerate(xgb_model.feature_importances_):
##    print(f"Dim {i}: {imp:.4f}")##

#----------------------------------------------------------------------
# backwardV0.py
#----------------------------------------------------------------------
##from linecache import cache
##
##import  numpy as np
##
##num_samples = 6
##num_features = 4
##num_hidden_dim = 7
##
### --- activation function ---
##def relu(x):
##    return np.maximum(0, x)
##
##def relu_derivative(x):
##    return (x > 0).astype(float)
##
### --- loss function ---
##def mse_loss(y_pred, y_true):
##    return np.mean((y_pred - y_true) ** 2)
##
##def mse_loss_derivative(y_pred, y_true):
##    return 2 * (y_pred - y_true) / y_true.size
##
##
### --- network parameters ---
##def initialize_parameters(n_input, n_hidden, n_output):
##    np.random.seed(42)
##    params = {
##        "W1": np.random.randn(n_hidden, n_input) * 0.01,
##        "b1": np.zeros((n_hidden, 1)),
##        "W2": np.random.randn(n_output, n_hidden) * 0.01,
##        "b2": np.zeros((n_output, 1)),
##    }
##    return params
##
### --- forward propagation ---
##def forward(X, params):
##    Z1 = params['W1'] @ X + params['b1']
##    A1 = relu(Z1)
##    Z2 = params['W2'] @ A1 + params['b2']
##    A2 = Z2
##
##    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
##    return A2, cache
##
### --- backward propagation ---
##def backward(y_true, cache, params):
##    m = y_true.shape[1]
##    dZ2 = mse_loss_derivative(cache['A2'], y_true)
##    dW2 = dZ2 @ cache['A1'].T / m
##    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
##
##    dA1 = params['W2'].T @ dZ2
##    dZ1 = dA1 * relu_derivative(cache["Z1"])
##    dW1 = dZ1 @ cache['X'].T / m
##    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
##
##    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
##    return grads
##
### --- parameter update ---
##def update_parameters(params, grads, learning_rate=0.1):
##    for key in params:
##        params[key] -= learning_rate * grads['d' + key]
##    return params
##
### --- training example ---
##if __name__ == "__main__":
##    # example data: 2 input features, 1 hidden layer (4 neurons), 1 ##output
##    #X = np.array([[0.5, -1.5], [1.0, 2.0]])
##    #y = np.array([[1.0, 0.0]])
##
##    # initialize parameters
##    #params = initialize_parameters(n_input=2, n_hidden=4, n_output=1)
##
##    #--------------------------------------------------------------------##--
##    X = np.random.rand(num_samples, num_features).T
##    y0 = np.random.rand(num_samples)
##    y1 = np.array([y0])
##
##    # initialize parameters
##    params = initialize_parameters(n_input=num_features, ##n_hidden=num_hidden_dim, n_output=1)
##
##    for idx in range(1000):
##        y_pred, cache = forward(X, params)
##        loss = mse_loss(y_pred, y1)
##        grads = backward(y1, cache, params)
##        params = update_parameters(params, grads, learning_rate=0.05)
##
##        if idx % 100 == 0:
##            print(f"step {idx}, loss: {loss:.4f}")

#----------------------------------------------------------------------
# batchnormV0.py
#----------------------------------------------------------------------

##import torch
##import torch.nn as nn
##
##class MyBatchNorm(nn.Module):
##    def __init__(self, num_features,
##                 eps:float = 1e-5,
##                 momentum: float = 0.1,
##                 affine: bool = True,
##                 track_running_stats: bool = True):
##        super().__init__()
##        self.eps = eps
##        self.momentum = momentum
##        self.affine = affine
##        self.tracking_running_statcks = track_running_stats
##
##        if self.affine:
##            self.weight = nn.Parameter(torch.ones(num_features))
##            self.bias = nn.Parameter(torch.zeros(num_features))
##        else:
##            self.register_parameter('weight', None)
##            self.register_parameter('bias', None)
##
##        if self.tracking_running_statcks:
##            self.register_buffer('running_mean', torch.zeros(##num_features))
##            self.register_buffer('running_var', torch.ones(num_features))
##            self.register_buffer('num_batches_tracked', torch.tensor(0, ##dtype=torch.long))
##        else:
##            self.register_parameter('running_mean', None)
##            self.register_parameter('running_var', None)
##            self.register_parameter('num_batches_tracked', None)
##
##    def forward(self, x:torch.Tensor) -> torch.Tensor:
##        # x shape: (N, C, *), we mormalize across N and spatial dims
##        if x.dim() < 2:
##            raise ValueError('Input must have at least 2 dims (N, C, ##...)')
##
##        #??? wrong
##        #if self.training or not self.tracking_running_statcks:
##        if self.training or self.tracking_running_statcks:
##            # leave channel dim(1) out
##            dims = (0,) + tuple(range(2, x.dim()))
##            batch_mean = x.mean(dim=dims, keepdim=True)
##            batch_var = x.var(dim=dims, unbiased=False, keepdim=True)
##
##            if self.tracking_running_statcks:
##                with torch.no_grad():
##                    self.running_mean = (1 - self.momentum) * ##self.running_mean \
##                                        + self.momentum * ##batch_mean.squeeze()
##                    self.running_var = (1 - self.momentum) * ##self.running_var \
##                                        + self.momentum * ##batch_var.squeeze()
##                    self.num_batches_tracked += 1
##
##                mean = batch_mean
##                var = batch_var
##            else:
##                #in eval() use running estimates
##                mean = self.running_mean.view(1, -1, *([1] * (x.dim() - ##2)))
##                var = self.running_var.view(1, -1, *([1] * (x.dim() - ##2)))
##
##            x_hat = (x-mean) / torch.sqrt(var + self.eps)
##
##            if self.affine:
##                w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
##                b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
##                x_hat = w * x_hat +b
##            return x_hat
##
##
##bn = MyBatchNorm(num_features=64)
##x = torch.randn(16, 64, 32, 32)
##out = bn(x)
###print("out = ", out)
##
##bn.eval()
##out_eval = bn(x)
###print("out_eval = ", out_eval)
##
##ref = nn.BatchNorm2d(64)
##custom = MyBatchNorm(64)
##custom.load_state_dict(ref.state_dict(), strict=False)
###print("custom.load_state_dict(ref.state_dict(), strict=False) = ",
### custom.load_state_dict(ref.state_dict(), strict=False))
##
##print("torch.isclose(out, out_eval, 1e-5, 1e-5) = ", torch.isclose(out, ##out_eval, 1e-5, 1e-5))
##print("(out != out_eval).sum() = ", (out != out_eval).sum())
##print("(out - out_eval).abs().max() = ", (out - out_eval).abs().max())


#----------------------------------------------------------------------
# layernormV0.py
#----------------------------------------------------------------------
### https://zhuanlan.zhihu.com/p/642043155
##import numpy as np
##import torch
##from torch import nn
##
##
##def torch_compare_fc(infeature, outfeature, bias, inputs, params, bias_params):
##    network = nn.Linear(infeature, outfeature, bias).requires_grad_(True)
##    cnt = 0
##    for i in network.parameters():
##        if cnt == 0:
##            i.data = torch.from_numpy(params.T)
##            i.retain_grad = True
##        else:
##            i.data = torch.from_numpy(bias_params)
##            i.retain_grad = True
##        cnt += 1
##
##    inputs = torch.tensor(inputs, requires_grad=True)
##    output = network(inputs)
##    sum = torch.sum(output)  # make sure the gradient is 1
##    kk = sum.backward()
##    grad_params = 0
##    grad_bias = 0
##    cnt = 0
##    for i in network.parameters():
##        if cnt == 0:
##            grad_params = i.grad
##        else:
##            grad_bias = i.grad
##        cnt += 1
##    inputs.retain_grad()
##    k = inputs.grad
##    return output, k, grad_params, grad_bias
##
##
##class fclayer(object):
##    def __init__(self, infeature, outfeature, bias=False, params=[], bias_params=[], ##name='', init=''):
##        self.infeature = infeature
##        self.outfeature = outfeature
##        self.bias = bias
##        if list(params) != []:
##            self.params = params
##        else:
##            ranges = np.sqrt(6 / (infeature + outfeature))
##            self.params = np.random.uniform(-ranges, ranges, (infeature, outfeature))
##        if bias and list(bias_params) != []:
##            self.bias_params = bias_params
##        else:
##            ranges = np.sqrt(6 / (infeature + outfeature))
##            self.bias_params = np.random.uniform(-ranges, ranges, (outfeature))
##        self.params_delta = np.zeros((infeature, outfeature))
##        self.bias_delta = np.zeros(outfeature)
##
##    def forward(self, inputs):
##        self.inputs = inputs
##        output = np.matmul(inputs, self.params)
##        if self.bias:
##            output = output + self.bias_params[np.newaxis, :]
##        return output
##
##    def backward(self, delta, lr=1e-10):
##        # previous layer delta
##        input_delta = np.matmul(delta, self.params.T)
##
##        # params bias delta
##        self.params_delta = np.matmul(delta.T, self.inputs).T
##        self.bias_delta = np.sum(delta, axis=0)
##
##        self.params -= self.params_delta * lr
##        if self.bias:
##            self.bias_params -= self.bias_delta * lr
##        return input_delta
##
##    def save_model(self):
##        return [self.params, self.bias_params]
##
##    def restore_model(self, models):
##        self.params = models[0]
##        self.bias_params = models[1]
##
##
##def train_single():
##    inputs = np.random.rand(3, 1000)
##    outputs = np.random.rand(3, 900)
##    infeature = inputs.shape[-1]
##    outfeature = 900
##    bias = True
##    delta = np.ones((inputs.shape[0], outfeature), dtype=np.float64)
##    params = np.random.standard_normal((infeature, outfeature)) / np.sqrt(infeature / 2)
##    if bias:
##        bias_params = np.random.standard_normal(outfeature) / np.sqrt(infeature / 2)
##
##    fc = fclayer(infeature, outfeature, bias, params, bias_params)
##    for i in range(1000):
##        out = fc.forward(inputs)
##        sum = np.sum((outputs - out) * (outputs - out))
##        delta = 2 * (out - outputs)
##        partial = fc.backward(delta, 0.0001)
##        print(sum)
##
##
##if __name__ == "__main__":
##    train_single()
##
##    inputs = np.random.rand(3, 1000)
##    infeature = inputs.shape[-1]
##    outfeature = 900
##    bias = True
##    delta = np.ones((inputs.shape[0], outfeature), dtype=np.float64)
##    params = np.random.standard_normal((infeature, outfeature)) / np.sqrt(infeature / 2)
##    if bias:
##        bias_params = np.random.standard_normal(outfeature) / np.sqrt(infeature / 2)
##
##    fc = fclayer(infeature, outfeature, bias, params, bias_params)
##    output = fc.forward(inputs)
##    partial = fc.backward(delta)
##    output_torch, partial_torch, grad_params_torch, grad_bias_torch = torch_compare_fc(##infeature, outfeature, bias, inputs, params, bias_params)
##    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(
##        np.abs(output - output_torch.cpu().detach().numpy()))
##    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-6, np.mean(
##        np.abs(partial - partial_torch.cpu().detach().numpy()))
##    assert np.mean(np.abs(fc.params_delta.T - grad_params_torch.cpu().detach().numpy())) < ##1e-6, np.mean(
##        np.abs(fc.params_delta.T - grad_params_torch.cpu().detach().numpy()))
##    assert np.mean(np.abs(fc.bias_delta - grad_bias_torch.cpu().detach().numpy())) < 1e-6, ##np.mean(
##        np.abs(fc.bias_delta - grad_bias_torch.cpu().detach().numpy()))##

#----------------------------------------------------------------------
#layernormV1.py
#----------------------------------------------------------------------
##import torch
##import torch.nn as nn
##
##class MyLayerNorm(nn.Module):
##    def __init__(self, normalizd_shape, eps=1e-5, affine=True):
##        super().__init__()
##        if isinstance(normalizd_shape, int):
##            normalizd_shape = (normalizd_shape,)
##        self.normalized_shape = tuple(normalizd_shape)
##        self.eps = eps
##        self.affine = affine
##
##        if self.affine:
##            # gamma, beta 's shape is the same as normalized_shape
##            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
##            self.bias   = nn.Parameter(torch.zeros(self.normalized_shape))
##        else:
##            self.register_parameter('weight', None)
##            self.register_parameter('bias', None)
##
##    def forward(self, x):
##        # x could be any shape, will normalize len(normalized_shape) at last
##        # for example, normalized_shape = (D,), will normalize the last dimension
##        dim = tuple(range(-len(self.normalized_shape), 0))
##        mean = x.mean(dim=dim, keepdim=True)
##        var = x.var(dim=dim, unbiased=False, keepdim=True)
##
##        x_hat = (x - mean) / torch.sqrt(var + self.eps)
##
##        if self.affine:
##            x_hat = x_hat * self.weight.view(*([1] * (x.dim() - len(##self.normalized_shape))),
##                                             *self.normalized_shape) \
##                          + self.bias.view(*([1] * (x.dim() - len(self.normalized_shape))),
##                                             *self.normalized_shape)
##        return x_hat
##
##if __name__ == '__main__':
##    # assume input is (batch_size, seq_len, feature_dim), then nor,alize at feature_dim
##    #ln = MyLayerNorm(normalizd_shape=128)
##    ln = MyLayerNorm(normalizd_shape=[10, 128])
##    x = torch.randn(32, 10, 128)
##    y = ln(x)
##    print(y.shape)##

#----------------------------------------------------------------------
#transformerBlockV0.py
#----------------------------------------------------------------------
##import torch
##import torch.nn as nn
##
##class TransformerBlock(nn.Module):
##    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
##        super(TransformerBlock, self).__init__()
##        self.attn = nn.MultiheadAttention(embed_dim, num_heads, ##dropout=dropout)
##        self.norm1 = nn.LayerNorm(embed_dim)
##        self.ff = nn.Sequential(
##            nn.Linear(embed_dim, ff_hidden_dim),
##            nn.ReLU(),
##            nn.Linear(ff_hidden_dim, embed_dim)
##        )
##        self.norm2 = nn.LayerNorm(embed_dim)
##        self.dropout = nn.Dropout(dropout)
##
##    def forward(self, x, attn_mask=None):
##        # --- self attention ---
##        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
##        x = x + self.dropout(attn_output)
##        x = self.norm1(x)
##
##        # --- feed forward---
##        ff_output = self.ff(x)
##        x = x + self.dropout(ff_output)
##        x = self.norm2(x)
##
##        return x
##
##seq_len = 10
##batch_size = 32
##embed_dim = 64
##num_heads = 8
##ff_hidden_dim = 256
##
##x = torch.randn(seq_len, batch_size, embed_dim)
##block = TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
##
##out = block(x)



#----------------------------------------------------------------------
#selfAttentionV0.py
#----------------------------------------------------------------------
##import torch
##import torch.nn as nn
##from torch.distributions.constraints import lower_triangular
##from torchtyping import TensorType
##import numpy as np
##
##class SingleHeadAttention(nn.Module):
##    def __init__(self, embedding_dim: int, attention_dim: int):
##        super().__init__()
##        torch.manual_seed(0)
##        self.key_gen = nn.Linear(embedding_dim, attention_dim, ##bias=False)
##        self.query_gen = nn.Linear(embedding_dim, attention_dim, ##bias=False)
##        self.value_gen = nn.Linear(embedding_dim, attention_dim, ##bias=False)
##
##    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
##        k = self.key_gen(embedded)
##        q = self.query_gen(embedded)
##        v = self.value_gen(embedded)
##
##        k_transposed = torch.transpose(k, 1, 2)
##        scores = torch.matmul(q, k_transposed)
##        batch_size = k.shape[0]
##        context_length = k.shape[1]
##        attention_dim = k.shape[2]
##
##        scores = scores / (attention_dim ** 0.5)
##
##        lower_triangle = torch.tril(torch.ones(context_length, ##context_length))
##        mask = lower_triangle == 0
##        scores = scores.masked_fill(mask, float("-Infinity"))
##        scores = nn.functional.softmax(scores, dim=2)
##
##        rst = scores @ v
##        rst = torch.round(rst, decimals=4)
##        return rst
##
##
##dim0 = 2
##dim1 = 3
##dim2 = 4
##
##embedded = np.random.rand(dim0, dim1, dim2).astype(np.float32)
##sol = SingleHeadAttention(dim2, dim1)
##
##embedded_tensor = torch.tensor(embedded)
##rst = sol.forward(embedded_tensor)
##print("rst = ", rst)


#----------------------------------------------------------------------
#pytorchVisualizerV0.py
#----------------------------------------------------------------------
##import torch
##from torch import nn
##
##torch.cuda.memory._record_memory_history(max_entries = 100000)
##torch.cuda.memory._record_memory_history()
##
##model = nn.Linear(10_000, 50_000, device='cuda')
##for _ in range(3):
##    inputs = torch.randn(5_000, 10_000, device='cuda')
##    outputs = model(inputs)
##
##torch.cuda.memory._dump_snapshot("//Users//shizhefu0//Desktop//tmp//profi##le.pkl")
##torch.cuda.memory._record_memory_history(enabled=None)



#----------------------------------------------------------------------
#kmeansV0.py
#----------------------------------------------------------------------
##import numpy as np
##import matplotlib.pyplot as plt
##from sklearn.datasets import make_blobs
##
##X, y = make_blobs(n_samples=500, n_features=2, centers=3, ##random_state=23)
##
##fig = plt.figure(0)
##plt.grid(True)
##plt.scatter(X[:,0], X[:,1])
##plt.show()
##
##k=3
##
##clusters = {}
##np.random.seed(23)
##
##for idx in range(k):
##    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
##    points = []
##    cluster = {
##        'center': center,
##        'points': []
##    }
##
##    clusters[idx] = cluster
##
##clusters
##
##plt.scatter(X[:, 0], X[:, 1])
##plt.grid(True)
##for idx in clusters:
##    center = clusters[idx]['center']
##    plt.scatter(center[0], center[1], marker='*', c='red')
##plt.show()
##
##def distance(p1, p2):
##    return np.sqrt(np.sum((p1-p2)**2))
##
##def assignClusters(X, clusters):
##    for idx in range(X.shape[0]):
##        dist = []
##        curr_x = X[idx]
##
##        for idx in range(k):
##            dis0 = distance(curr_x, clusters[idx]['center'])
##            dist.append(dis0)
##
##        curr_cluster = np.argmin(dist)
##        clusters[curr_cluster]['points'].append(curr_x)
##
##    return clusters
##
##
##def updateClusters(X, clusters):
##    for idx in range(k):
##        points = np.array(clusters[idx]['points'])
##        if points.shape[0] > 0:
##            new_center = points.mean(axis=0)
##            clusters[idx]['center'] = new_center
##            clusters[idx]['points'] = []
##
##    return clusters
##
##def predCluster(X, clusters):
##    pred = []
##    for idx0 in range(X.shape[0]):
##        dist = []
##        for idx1 in range(k):
##            tmp = distance(X[idx], clusters[idx1]['center'])
##            dist.append(tmp)
##
##        pred.append(np.argmin(dist))
##
##    return pred
##
##clusters = assignClusters(X, clusters)
##clusters = updateClusters(X, clusters)
##pred = predCluster(X, clusters)
##
##plt.scatter(X[:,0], X[:, 1], c=pred)
##for idx in clusters:
##    center = clusters[idx]['center']
##    plt.scatter(center[0], center[1], marker='^', c='red')
##plt.show()##


#----------------------------------------------------------------------
#kmeansV1.py
#----------------------------------------------------------------------
##import numpy as np
##import pandas as pd
##import matplotlib.pyplot as plt
##from matplotlib.colors import ListedColormap
###%matplotlib inline
##
##blobs = pd.read_csv('//Users//shizhefu0//Desktop//ml//data//other//kmean##s_blobs.csv')
##colnames = list(blobs.columns[1:-1])
##blobs.head()
##
##customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta"])
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(x=blobs['x'], y=blobs['y'], s=150,
##            c=blobs['cluster'].astype('category'),
##            cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##def initiate_centroids(k, dset):
##    '''
##    Select k data points as centroids
##    k: number of centroids
##    dset: pandas dataframe
##    '''
##    centroids = dset.sample(k)
##    return centroids
##
##np.random.seed(42)
##k=3
##df = blobs[['x','y']]
##centroids = initiate_centroids(k, df)
##centroids
##
##def rsserr(a,b):
##    '''
##    Calculate the root of sum of squared errors.
##    a and b are numpy arrays
##    '''
##    return np.square(np.sum((a-b)**2))
##
##
##for i, centroid in enumerate(range(centroids.shape[0])):
##    err = rsserr(centroids.iloc[centroid,:], df.iloc[36,:])
##    print('Error for centroid {0}: {1:.2f}'.format(i, err))
##
##
##def centroid_assignation(dset, centroids):
##    '''
##    Given a dataframe `dset` and a set of `centroids`, we assign each
##    data point in `dset` to a centroid.
##    - dset - pandas dataframe with observations
##    - centroids - pa das dataframe with centroids
##    '''
##    k = centroids.shape[0]
##    n = dset.shape[0]
##    assignation = []
##    assign_errors = []
##
##    for obs in range(n):
##        # Estimate error
##        all_errors = np.array([])
##        for centroid in range(k):
##            err = rsserr(centroids.iloc[centroid, :], dset.iloc[obs,:])
##            all_errors = np.append(all_errors, err)
##
##        # Get the nearest centroid and the error
##        nearest_centroid =  np.where(all_errors==np.amin(##all_errors))[0].tolist()[0]
##        nearest_centroid_error = np.amin(all_errors)
##
##        # Add values to corresponding lists
##        assignation.append(nearest_centroid)
##        assign_errors.append(nearest_centroid_error)
##
##    return assignation, assign_errors
##
##
##df['centroid'], df['error'] = centroid_assignation(df, centroids)
##df.head()
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o',
##            c=df['centroid'].astype('category'),
##            cmap = customcmap, s=80, alpha=0.5)
##plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],
##            marker = 's', s=200, c=[0, 1, 2],
##            cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##print("The total error is {0:.2f}".format(df['error'].sum()))
##
##centroids = df.groupby('centroid').agg('mean').loc[:, ##colnames].reset_index(drop = True)
##centroids
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o',
##            c=df['centroid'].astype('category'),
##            cmap = customcmap, s=80, alpha=0.5)
##plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],
##            marker = 's', s=200,
##            c=[0, 1, 2], cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##
##def kmeans(dset, k=2, tol=1e-4):
##    '''
##    K-means implementationd for a
##    `dset`:  DataFrame with observations
##    `k`: number of clusters, default k=2
##    `tol`: tolerance=1E-4
##    '''
##    # Let us work in a copy, so we don't mess the original
##    working_dset = dset.copy()
##    # We define some variables to hold the error, the
##    # stopping signal and a counter for the iterations
##    err = []
##    goahead = True
##    j = 0
##
##    # Step 2: Initiate clusters by defining centroids
##    centroids = initiate_centroids(k, dset)
##
##    while (goahead):
##        # Step 3 and 4 - Assign centroids and calculate error
##        working_dset['centroid'], j_err = centroid_assignation(##working_dset, centroids)
##        err.append(sum(j_err))
##
##        # Step 5 - Update centroid position
##        centroids = ##working_dset.groupby('centroid').agg('mean').reset_index(##drop=True)
##
##        # Step 6 - Restart the iteration
##        if j > 0:
##            # Is the error less than a tolerance (1E-4)
##            if err[j - 1] - err[j] <= tol:
##                goahead = False
##        j += 1
##
##    working_dset['centroid'], j_err = centroid_assignation(##working_dset, centroids)
##    centroids = working_dset.groupby('centroid').agg('mean').reset_index##(drop=True)
##    return working_dset['centroid'], j_err, centroids
##
##
##np.random.seed(42)
##df['centroid'], df['error'], centroids =  kmeans(df[['x','y']], 3)
##df.head()
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o',
##            c=df['centroid'].astype('category'),
##            cmap = customcmap, s=80, alpha=0.5)
##plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],
##            marker = 's', s=200, c=[0, 1, 2],
##            cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##
##err_total = []
##n = 10
##
##df_elbow = blobs[['x','y']]
##
##for i in range(n):
##    _, my_errs, _ = kmeans(df_elbow, i+1)
##    err_total.append(sum(my_errs))
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.plot(range(1,n+1), err_total, linewidth=3, marker='o')
##ax.set_xlabel(r'Number of clusters', fontsize=14)
##ax.set_ylabel(r'Total error', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##
###------------------
###using scikit learn
###------------------
##
####from sklearn.cluster import KMeans
####from sklearn import datasets
####from sklearn.utils import shuffle
####
##### import some data to play with
####iris = datasets.load_iris()
####X = iris.data
####y = iris.target
####names = iris.feature_names
####X, y = shuffle(X, y, random_state=42)
####
####model = KMeans(n_clusters=3, random_state=42)
####iris_kmeans = model.fit(X)
####
####iris_kmeans.labels_
####
####y = np.choose(y, [1, 2, 0]).astype(int)
####y
####
####from sklearn.metrics import confusion_matrix
####
####conf_matrix = confusion_matrix(y, iris_kmeans.labels_)
####
####fig, ax = plt.subplots(figsize=(7.5, 7.5))
####ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
####for i in range(conf_matrix.shape[0]):
####    for j in range(conf_matrix.shape[1]):
####        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center',
####                ha='center', size='xx-large')
####
####plt.xlabel('Predictions', fontsize=18)
####plt.ylabel('Actuals', fontsize=18)
####plt.title('Confusion Matrix', fontsize=18)
####plt.show()
####
####iris_kmeans.cluster_centers_
####
####fig = plt.figure(figsize=(20, 10))
####ax1 = fig.add_subplot(1, 2, 1, projection='3d')
####ax1.scatter(X[:, 3], X[:, 0], X[:, 2],
####            c=iris_kmeans.labels_.astype(float),
####           edgecolor="k", s=150, cmap=customcmap)
####ax1.view_init(20, -50)
####ax1.set_xlabel(names[3], fontsize=12)
####ax1.set_ylabel(names[0], fontsize=12)
####ax1.set_zlabel(names[2], fontsize=12)
####ax1.set_title("K-Means Clusters for the Iris Dataset", fontsize=12)
####
####ax2 = fig.add_subplot(1, 2, 2, projection='3d')
####
####for label, name in enumerate(['virginica','setosa','versicolor']):
####    ax2.text3D(
####        X[y == label, 3].mean(),
####        X[y == label, 0].mean(),
####        X[y == label, 2].mean() + 2,
####        name,
####        horizontalalignment="center",
####        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
####    )
####
####ax2.scatter(X[:, 3], X[:, 0], X[:, 2],
####            c=y, edgecolor="k", s=150,
####            cmap=customcmap)
####ax2.view_init(20, -50)
####ax2.set_xlabel(names[3], fontsize=12)
####ax2.set_ylabel(names[0], fontsize=12)
####ax2.set_zlabel(names[2], fontsize=12)
####ax2.set_title("Actual Labels for the Iris Dataset", fontsize=12)
####fig.show()




#----------------------------------------------------------------------
#kmeansV2.py
#----------------------------------------------------------------------
##import numpy as np
##import pandas as pd
##import matplotlib.pyplot as plt
##from matplotlib.colors import ListedColormap
###%matplotlib inline
##
##blobs = pd.read_csv('//Users//shizhefu0//Desktop//ml//data//other//kmean##s_blobs.csv')
##colnames = list(blobs.columns[1:-1])
##blobs.head()
##
##customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta"])
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(x=blobs['x'], y=blobs['y'], s=150,
##            c=blobs['cluster'].astype('category'),
##            cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##def initiateCentroids(k, dset):
##    '''
##    Select k data points as centroids
##    k: number of centroids
##    dset: pandas dataframe
##    '''
##    centroids = dset.sample(k)
##    return centroids
##
##np.random.seed(42)
##k=3
##df = blobs[['x','y']]
##centroids = initiateCentroids(k, df)
##centroids
##
##def rsserr(a,b):
##    '''
##    Calculate the root of sum of squared errors.
##    a and b are numpy arrays
##    '''
##    return np.square(np.sum((a-b)**2))
##
##
##for i, centroid in enumerate(range(centroids.shape[0])):
##    err = rsserr(centroids.iloc[centroid,:], df.iloc[36,:])
##    print('Error for centroid {0}: {1:.2f}'.format(i, err))
##
##
##def centroidAssignation(dset, centroids):
##    '''
##    Given a dataframe `dset` and a set of `centroids`, we assign each
##    data point in `dset` to a centroid.
##    - dset - pandas dataframe with observations
##    - centroids - pa das dataframe with centroids
##    '''
##    k = centroids.shape[0]
##    n = dset.shape[0]
##    assignation = []
##    assign_errors = []
##
##    for obs in range(n):
##        # Estimate error
##        all_errors = []
##        for centroid in range(k):
##            err = rsserr(centroids.iloc[centroid, :], dset.iloc[obs,:])
##            all_errors = np.append(all_errors, err)
##
##        # Get the nearest centroid and the error
##        sorted_index =  np.argsort(all_errors)
##        min_index = sorted_index[0]
##        nearest_centroid = min_index
##        nearest_centroid_error = all_errors[min_index]
##
##        assignation.append(nearest_centroid)
##        assign_errors.append(nearest_centroid_error)
##
##    return assignation, assign_errors
##
##
##df['centroid'], df['error'] = centroidAssignation(df, centroids)
##df.head()
##
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o',
##            c=df['centroid'].astype('category'),
##            cmap = customcmap, s=80, alpha=0.5)
##plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],
##            marker = 's', s=200, c=[0, 1, 2],
##            cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##print("The total error is {0:.2f}".format(df['error'].sum()))
##
##centroids = df.groupby('centroid').agg('mean').loc[:, ##colnames].reset_index(drop = True)
##centroids
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o',
##            c=df['centroid'].astype('category'),
##            cmap = customcmap, s=80, alpha=0.5)
##plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],
##            marker = 's', s=200,
##            c=[0, 1, 2], cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##
##def kmeans(dset, k=2, tol=1e-4):
##    '''
##    K-means implementationd for a
##    `dset`:  DataFrame with observations
##    `k`: number of clusters, default k=2
##    `tol`: tolerance=1E-4
##    '''
##    # Let us work in a copy, so we don't mess the original
##    working_dset = dset.copy()
##    # We define some variables to hold the error, the
##    # stopping signal and a counter for the iterations
##    err = []
##    goahead = True
##    j = 0
##
##    # Step 2: Initiate clusters by defining centroids
##    centroids = initiateCentroids(k, dset)
##
##    while (goahead):
##        # Step 3 and 4 - Assign centroids and calculate error
##        working_dset['centroid'], j_err = centroidAssignation(##working_dset, centroids)
##        err.append(sum(j_err))
##
##        # Step 5 - Update centroid position
##        centroids = ##working_dset.groupby('centroid').agg('mean').reset_index(##drop=True)
##
##        # Step 6 - Restart the iteration
##        if j > 0:
##            # Is the error less than a tolerance (1E-4)
##            if err[j - 1] - err[j] <= tol:
##                goahead = False
##        j += 1
##
##    working_dset['centroid'], j_err = centroidAssignation(working_dset, ##centroids)
##    centroids = working_dset.groupby('centroid').agg('mean').reset_index##(drop=True)
##    return working_dset['centroid'], j_err, centroids
##
##
##np.random.seed(42)
##df['centroid'], df['error'], centroids =  kmeans(df[['x','y']], 3)
##df.head()
##
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o',
##            c=df['centroid'].astype('category'),
##            cmap = customcmap, s=80, alpha=0.5)
##plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],
##            marker = 's', s=200, c=[0, 1, 2],
##            cmap = customcmap)
##ax.set_xlabel(r'x', fontsize=14)
##ax.set_ylabel(r'y', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()
##
##
##err_total = []
##n = 10
##
##df_elbow = blobs[['x','y']]
##
##for i in range(n):
##    _, my_errs, _ = kmeans(df_elbow, i+1)
##    err_total.append(sum(my_errs))
##fig, ax = plt.subplots(figsize=(8, 6))
##plt.plot(range(1,n+1), err_total, linewidth=3, marker='o')
##ax.set_xlabel(r'Number of clusters', fontsize=14)
##ax.set_ylabel(r'Total error', fontsize=14)
##plt.xticks(fontsize=12)
##plt.yticks(fontsize=12)
##plt.show()




#----------------------------------------------------------------------
#amazonReviewSentimentAnalysisV0.py
#----------------------------------------------------------------------
##import sys
##sys.path.append("//Users//shizhefu0//Desktop//ml//code//github_jeffjeff4##")
##print(sys.path)
##import pandasHelperV0 as pdh
##
##from pandas.core.common import random_state
##from sklearn.ensemble import RandomForestRegressor
##from tensorflow import double
##from tensorflow.python.feature_column.feature_column import linear_model
##from tensorflow.python.ops.numpy_ops.np_dtypes import float64
##
###from walmartSalesForecastingV0 import num_rows_before, numerical_cols
##
##
##import string
##from lib2to3.btm_utils import tokens
##
##import pandas as pd
##import numpy as np
##import seaborn as sns
##import matplotlib.pyplot as plt
##from nltk import pad_sequence
##from sklearn.cluster import KMeans
##import multiprocessing as mp
##import gc
##import datetime
##from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##import calendar
##from scipy.sparse import csr_matrix,hstack
##import tensorflow as tf
##from sklearn.linear_model import LinearRegression
##from sklearn.feature_extraction.text import CountVectorizer
##from sklearn.metrics import mean_squared_error
##from tensorflow.python.ops.gen_array_ops import upper_bound
###from lightgbm import LGBMRegressor
##from tqdm import tqdm
##import pickle
##from scipy import stats
##
##import nltk
##from nltk.stem import WordNetLemmatizer
##wordnet_lemmatizer = WordNetLemmatizer()
##
##import string
##
##from collections import Counter
##import re
##
##import torch
##import torch.nn as nn
##from torch.utils.data import TensorDataset, DataLoader
##
##from sklearn import linear_model
##from sklearn.ensemble import RandomForestRegressor
##import xgboost as xgb
###import catboost as cb
##import lightgbm as lgb
##from sklearn.metrics import mean_squared_error
##from sklearn.preprocessing import LabelEncoder
##
##import time
##
###------------------------------------------
###
###------------------------------------------
##
##def splitXY(corpus):
##    print('Splitting...')
##    start = time.time()
##    X = []
##    y = []
##
##    for idx in corpus:
##        X.append(idx.split(' ', 1)[1])
##        if int(list(idx.split(' ', 1)[0])[9]) == 1:
##            y.append(0)
##        else:
##            y.append(1)
##
##    y = np.array(y)
##    print('It took', time.time() - start, 'sec to split.')
##    return X, y
##
##
##file_path = ''
##corpus = pdh.loadDataset()
##corpus = corpus[:100]
##
##X, y = splitXY(corpus)
##y = torch.tensor(y, dtype=torch.long)
##
##clean_x = []
##for sentence in X:
##    tokens = pdh.cleanAndTokenize(sentence)
##    sent = ' '.join(tokens)
##    clean_x.append(sent)
##
##
###------------------------------------------
###
###------------------------------------------
##
##vocab = pdh.buildVocab(clean_x)
##X_tensor = pdh.sentencesToTensor(clean_x, vocab, max_len=8)
##
##print(f'vocabulary: {vocab}')
##print(f'input tensor: {X_tensor}')
##
##seq_len = 10
##vocab_size = 1000
##embed_dim = 32
##num_heads = 4
##ff_dim = 64
##num_layers = 2
##batch_size = 8
##num_classes = 2
##lr = 1e-3
##epochs = 10
##
### --- initiate model ---
##model = pdh.TransformerClassifier(vocab_size, embed_dim, num_heads, ##ff_dim,
##                                  num_layers, num_classes)
##
### --- run training ---
##for epoch in range(epochs):
##    loss = pdh.train(model, X_tensor, y)
##    if epoch % 2 == 0:
##        print(f'epoch {epoch}: loss = {loss:.4f}')
##
### --- predict ---
##preds = pdh.predict(model, X_tensor)
##print(f"prediction: {preds.tolist()}")
##print(f"ground truth: {y.tolist()}")
##
##from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
###cm = confusion_matrix(y, preds, labels=model.classes_)
##cm = confusion_matrix(y, preds)
##
##disp = ConfusionMatrixDisplay(confusion_matrix=cm)
##disp.plot(cmap = plt.cm.Blues)
##plt.title('confusion matrix for order priority prediction')
##plt.show()
##
##print("")



#----------------------------------------------------------------------
#linearRegressionV2.py
#----------------------------------------------------------------------
### Importing libraries
##
##import numpy as np
##
##import pandas as pd
##
##from sklearn.model_selection import train_test_split
##
##import matplotlib.pyplot as plt
##
##
### Linear Regression
##
##class LinearRegression():
##
##    def __init__(self, learning_rate, iterations):
##        self.learning_rate = learning_rate
##
##        self.iterations = iterations
##
##    # Function for model training
##
##    def fit(self, X, Y):
##        # no_of_training_examples, no_of_features
##
##        self.m, self.n = X.shape
##
##        # weight initialization
##
##        self.W = np.zeros(self.n)
##
##        self.b = 0
##
##        self.X = X
##
##        self.Y = Y
##
##        # gradient descent learning
##
##        for i in range(self.iterations):
##            self.update_weights()
##
##        return self
##
##    # Helper function to update weights in gradient descent
##
##    def update_weights(self):
##        Y_pred = self.predict(self.X)
##
##        # calculate gradients
##
##        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
##
##        db = - 2 * np.sum(self.Y - Y_pred) / self.m
##
##        # update weights
##
##        self.W = self.W - self.learning_rate * dW
##
##        self.b = self.b - self.learning_rate * db
##
##        return self
##
##    # Hypothetical function  h( x )
##
##    def predict(self, X):
##        return X.dot(self.W) + self.b
##
##
### driver code
##
##def main():
##    # Importing dataset
##
##    df = pd.read_csv("//Users//shizhefu0//Desktop//ml//data//Dataset-mai##n//Salary Data.csv")
##
##    X = df.iloc[:, :-1].values
##
##    Y = df.iloc[:, 1].values
##
##    # Splitting dataset into train and test set
##
##    X_train, X_test, Y_train, Y_test = train_test_split(
##        X, Y, test_size=1 / 3, random_state=0)
##
##    # Model training
##
##    model = LinearRegression(iterations=1000, learning_rate=0.01)
##
##    model.fit(X_train, Y_train)
##
##    # Prediction on test set
##
##    Y_pred = model.predict(X_test)
##
##    print("Predicted values ", np.round(Y_pred[:3], 2))
##
##    print("Real values      ", Y_test[:3])
##
##    print("Trained W        ", round(model.W[0], 2))
##
##    print("Trained b        ", round(model.b, 2))
##
##    # Visualization on test set
##
##    plt.scatter(X_test, Y_test, color='blue')
##
##    plt.plot(X_test, Y_pred, color='orange')
##
##    plt.title('Salary vs Experience')
##
##    plt.xlabel('Years of Experience')
##
##    plt.ylabel('Salary')
##
##    plt.show()
##
##
##if __name__ == "__main__":
##    main()##

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


