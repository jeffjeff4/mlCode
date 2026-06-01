import sys
sys.path.append("//Users//shizhefu0//Desktop//ml//code//github_jeffjeff4")
print(sys.path)
import pandasHelperV0 as pdh

from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor
from tensorflow import double
from tensorflow.python.feature_column.feature_column import linear_model
from tensorflow.python.ops.numpy_ops.np_dtypes import float64

#from walmartSalesForecastingV0 import num_rows_before, numerical_cols


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

import time

#------------------------------------------
#
#------------------------------------------

def splitXY(corpus):
    print('Splitting...')
    start = time.time()
    X = []
    y = []

    for idx in corpus:
        X.append(idx.split(' ', 1)[1])
        if int(list(idx.split(' ', 1)[0])[9]) == 1:
            y.append(0)
        else:
            y.append(1)

    y = np.array(y)
    print('It took', time.time() - start, 'sec to split.')
    return X, y


file_path = ''
corpus = pdh.loadDataset()
corpus = corpus[:100]

X, y = splitXY(corpus)
y = torch.tensor(y, dtype=torch.long)

clean_x = []
for sentence in X:
    tokens = pdh.cleanAndTokenize(sentence)
    sent = ' '.join(tokens)
    clean_x.append(sent)


#------------------------------------------
#
#------------------------------------------

vocab = pdh.buildVocab(clean_x)
X_tensor = pdh.sentencesToTensor(clean_x, vocab, max_len=8)

print(f'vocabulary: {vocab}')
print(f'input tensor: {X_tensor}')

seq_len = 10
vocab_size = 1000
embed_dim = 32
num_heads = 4
ff_dim = 64
num_layers = 2
batch_size = 8
num_classes = 2
lr = 1e-3
epochs = 10

# --- initiate model ---
model = pdh.TransformerClassifier(vocab_size, embed_dim, num_heads, ff_dim,
                                  num_layers, num_classes)

# --- run training ---
for epoch in range(epochs):
    loss = pdh.train(model, X_tensor, y)
    if epoch % 2 == 0:
        print(f'epoch {epoch}: loss = {loss:.4f}')

# --- predict ---
tmp_preds = pdh.predict(model, X_tensor)
preds = torch.argmax(tmp_preds, axis=1)
print(f"prediction: {preds.tolist()}")
print(f"ground truth: {y.tolist()}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#cm = confusion_matrix(y, preds, labels=model.classes_)
cm = confusion_matrix(y, preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap = plt.cm.Blues)
plt.title('confusion matrix for order priority prediction')
plt.show()

print("")

