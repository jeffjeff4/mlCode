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

#****************************
#manual code
#****************************

#col_name = 'Rating'
#new_col_name = 'Class'
#training_df = review_clean_df
#training_df.loc[training_df[col_name].isin([1.0, 2.0]), new_col_name] = 'Bad'
#training_df.loc[training_df[col_name].isin([3.0]), new_col_name] = 'Neutral'
#training_df.loc[training_df[col_name].isin([4.0, 5.0]), new_col_name] = 'Good'


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
def getHeatMapv0(df, df_name, is_numeric_only=True, need_plot=True):
    print("{0}".format(df_name))
    corr = df.corr(numeric_only=is_numeric_only)
    getDatasetInfo(corr, "corr")
    if need_plot == True:
        sns.heatmap(corr, annot=True, fmt=".2f", square=True, linewidths=.5)
        plt.show()

def drawBoxplot(df, col_name, fig_size_a=8, fig_size_b=6, need_plot=True):
    plt.figure(figsize=(fig_size_a, fig_size_a))
    sns.boxplot(x=df[col_name])
    plt.title('Boxplot for {0}'.format(col_name))
    if need_plot == True:
        plt.show()

def drawScatterplot(df, col_name_x, col_name_y, fig_size_a=8, fig_size_b=6, need_plot=True):
    plt.figure(figsize=(fig_size_a, fig_size_a))
    sns.regplot(x=df[col_name_x], y=df[col_name_y], scatter_kws={'alpha': 0.5}, line_kws={"color": "darkblue"})
    plt.title('Scatter plot - {0} vs {1} with regression line'.format(col_name_x, col_name_y))
    plt.xlabel('{0}'.format(col_name_x))
    plt.ylabel('{0}'.format(col_name_y))

    if need_plot == True:
        plt.show()

def drawHistogram(df, col_name, fig_size_a=8, fig_size_b=6, this_color='g', num_of_bins=100, need_plot=True):
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


#-----------------------------------------------------------------------
# nlp data processing
#-----------------------------------------------------------------------
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


def build_vocab(texts, min_freq=1, special_tokens=['<pad>', '<unk>']):
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
        output = output.mean(dim=1)  # Average over sequence
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




