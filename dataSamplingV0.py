##方法选择指南
##欠采样：当多数类数据量极大时使用，可能丢失重要信息
##过采样：当少数类数据量较少时使用，可能导致过拟合
##SMOTE/ADASYN：适合特征空间连续的情况，对离散特征效果不佳
##负采样：推荐系统/NLP中处理负样本过多的问题
##分层采样：保持数据分布，特别适用于评估模型性能
##根据数据集大小、不平衡程度和具体任务需求选择合适的方法，通常需要实验验证哪种方法效果最好。


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

NUM_SAMPLES = 100

# 创建不平衡数据集
X, y = make_classification(n_samples=NUM_SAMPLES, weights=[0.9, 0.1], random_state=42)
print("原始分布:", Counter(y))

# 随机欠采样
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
print("欠采样后分布:", Counter(y_res))

import numpy as np
import pandas as pd

# 创建DataFrame示例
df = pd.DataFrame({
    'feature1': np.random.randn(NUM_SAMPLES),
    'feature2': np.random.randn(NUM_SAMPLES),
    'label': [0] * 90 + [1] * 10  # 90%负样本，10%正样本
})


# 随机欠采样实现
def random_undersample(df, label_col, ratio=1.0):
    # ratio表示少数类与多数类的比例
    minority = df[df[label_col] == 1]
    majority = df[df[label_col] == 0]

    # 按比例采样多数类
    n_samples = int(len(minority) * ratio)
    majority_sampled = majority.sample(n_samples, random_state=42)

    return pd.concat([minority, majority_sampled])


balanced_df = random_undersample(df, 'label', ratio=1)
print("欠采样后标签分布:", balanced_df['label'].value_counts())


def random_oversample(df, label_col):
    minority = df[df[label_col] == 1]
    majority = df[df[label_col] == 0]

    # 过采样少数类
    minority_oversampled = minority.sample(len(majority), replace=True, random_state=42)

    return pd.concat([majority, minority_oversampled])


balanced_df = random_oversample(df, 'label')
print("过采样后标签分布:", balanced_df['label'].value_counts())

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("SMOTE后分布:", Counter(y_res))

import numpy as np


def negative_sampling(items, positive_items, n_negatives=5):
    """
    Improved negative sampling function

    Parameters:
    -----------
    items : list
        All candidate items
    positive_items : list
        List of positive items (can be single items or lists of items)
    n_negatives : int
        Number of negative samples per positive sample

    Returns:
    --------
    list
        List of negative samples
    """
    negatives = []

    # Convert single item to list if needed
    if isinstance(positive_items, (int, str)):
        positive_items = [positive_items]

    # Get all possible negative candidates
    negative_candidates = [item for item in items if item not in positive_items]

    # Sample negatives
    if len(negative_candidates) > 0:
        # Ensure we don't request more samples than available
        n_samples = min(n_negatives, len(negative_candidates))
        sampled = np.random.choice(negative_candidates, size=n_samples, replace=False)
        negatives.extend(sampled)

    return negatives

# 示例使用
all_items = list(range(NUM_SAMPLES))
positive_items = [3, 12, 25, 44, 68]  # 已知正样本
negative_samples = negative_sampling(all_items, positive_items, n_negatives=3)
print("负采样结果:", negative_samples)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("训练集分布:", Counter(y_train))
print("测试集分布:", Counter(y_test))

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Fold分布 - 训练:", Counter(y_train), "测试:", Counter(y_test))

def stratified_sample(df, stratify_col, frac=0.1):
    """
    df: 输入DataFrame
    stratify_col: 分层依据的列名
    frac: 采样比例
    """
    return df.groupby(stratify_col).apply(lambda x: x.sample(frac=frac)).reset_index(drop=True)

# 示例使用
sample_df = stratified_sample(df, 'label', frac=0.5)
print("分层采样结果分布:", sample_df['label'].value_counts())


import numpy as np


def stratified_sampleV1(df, strata_col, sample_size):
    """Manual stratified sampling implementation"""
    strata = df[strata_col].unique()
    sample = pd.DataFrame()

    for stratum in strata:
        stratum_data = df[df[strata_col] == stratum]
        stratum_sample = stratum_data.sample(frac=sample_size / len(df), random_state=42)
        sample = pd.concat([sample, stratum_sample])

    return sample


# Usage
feature_name = 'feature1'
sample = stratified_sampleV1(df, feature_name, 20)
print("\nManual stratified sample:\n", sample[feature_name].value_counts())


