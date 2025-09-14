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

#--------------------------------
# change numerical features to bins, then to embedding
#--------------------------------
class BinnedEmbedding(nn.Module):
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
'''
x_binned = torch.tensor([
 [1,4,2],
 [0,3,7],
 [2,5,1],
 ], dtype=torch.long)
'''

x_binned = torch.tensor([
 [1,4,2],
 [0,3,1],
 [2,5,0],
 [1,1,1],
 [0,2,2],
 ], dtype=torch.long) # shape(5,3)

# the bin number of each feature is 10, the output embedding dim is 4
model = BinnedEmbedding(num_bins_list=[10, 10, 10], embedding_dim=4)
output = model(x_binned)
print(output.shape) # output: torch.Size([3, 12]), 3 features, 4 dim size each
