import sys
sys.path.append("//Users//shizhefu0//Desktop//ml//code//github_jeffjeff4")
print(sys.path)
import pandasHelperV0 as pdh

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from pandas.api.types import is_numeric_dtype

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
num_samples = 25

# 1. Numerical features (with outliers and missing values)
numerical_data = {
    'sales_volume': np.random.lognormal(3, 1, num_samples).round(2),
    'price': np.random.uniform(10, 200, num_samples).round(2),
    'customer_age': np.random.randint(18, 70, num_samples).astype(float),  # Convert to float first
    'page_views': np.random.poisson(50, num_samples).astype(float),  # Convert to float first
    'discount_rate': np.random.uniform(0, 0.5, num_samples).round(2)
}

# Add outliers
numerical_data['sales_volume'][3] = 1000  # Extreme outlier
numerical_data['price'][7] = 500  # Extreme outlier
numerical_data['customer_age'][12] = 120  # Impossible value

# Add missing values (10-20% missing)
for col in numerical_data:
    mask = np.random.rand(num_samples) < 0.15
    numerical_data[col][mask] = np.nan

# Convert back to int where appropriate (after adding NaNs)
numerical_data['customer_age'] = pd.Series(numerical_data['customer_age']).astype('Int64')  # Pandas nullable integer
numerical_data['page_views'] = pd.Series(numerical_data['page_views']).astype('Int64')  # Pandas nullable integer

# 2. Categorical features (with missing values)
categorical_data = {
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', np.nan], num_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West', np.nan], num_samples),
    'customer_segment': np.random.choice(['New', 'Returning', 'VIP', 'Inactive', np.nan], num_samples),
    'payment_method': np.random.choice(['Credit', 'Debit', 'PayPal', 'Bank Transfer', np.nan], num_samples)
}

# 3. Embeddings (5-dimensional for example)
embedding_data = {
    'product_embedding': [np.random.randn(5) for _ in range(num_samples)],
    'user_embedding': [np.random.randn(5) for _ in range(num_samples)]
}

# Normalize embeddings
embedding_data['product_embedding'] = list(normalize(np.array(embedding_data['product_embedding'])))
embedding_data['user_embedding'] = list(normalize(np.array(embedding_data['user_embedding'])))

# Create correlated numerical features
numerical_data['sales_value'] = numerical_data['sales_volume'] * numerical_data['price'] * (1 - numerical_data['discount_rate'])
numerical_data['log_page_views'] = np.log(numerical_data['page_views'].astype(float) + 1)  # Handle NaN with float

# Create uncorrelated numerical feature
numerical_data['random_feature'] = np.random.rand(num_samples)

# Create DataFrame
df = pd.DataFrame({
    **numerical_data,
    **categorical_data,
    **embedding_data
})

# Add some correlation between categorical and numerical
df.loc[df['product_category'] == 'Electronics', 'price'] = df.loc[df['product_category'] == 'Electronics', 'price'] * 1.5
df.loc[df['customer_segment'] == 'VIP', 'sales_volume'] = df.loc[df['customer_segment'] == 'VIP', 'sales_volume'] * 2

# print the DataFrame
print("Generated DataFrame with mixed data types:")
print(df.head())

print("-------------------------------------")
print("")

# Show correlation matrix for numerical features
print("\nCorrelation matrix for numerical features:")
print(df[['sales_volume', 'price', 'customer_age', 'page_views', 'discount_rate', 'sales_value', 'log_page_views']].corr())

print("-------------------------------------")
print("")

# Show missing values summary
print("\nMissing values summary:")
print(df.isna().sum())

print("-------------------------------------")
print("")

#including

numerical_cols0 = [col for col in df.columns if is_numeric_dtype(df[col])]
df_numerical0 = df[numerical_cols0]
print("df_numerical0 = ")
print(df_numerical0)

print("-------------------------------------")
print("")

#excluding
numerical_cols1 = df.select_dtypes(exclude=['object', 'category', 'bool', 'datetime']).columns
df_numerical1 = df[numerical_cols1]
print("df_numerical1 = ")
print(df_numerical1)

print("-------------------------------------")
print("")

num_col2 = ['sales_volume', 'price']
print(df[num_col2].describe())

print("-------------------------------------")
print("")

cat_cols0 = ['product_category', 'region']
print(df[cat_cols0].describe(include='object'))

print("-------------------------------------")
print("")

# Custom statistics for selected columns
stats = df[['sales_volume', 'price']].agg(['mean', 'median', 'std', 'min', 'max', 'skew'])
print(stats)

print("-------------------------------------")
print("")

# Show value distributions
for col in ['product_category', 'customer_segment']:
    print(f"\n{col} value counts:")
    print(df[col].value_counts(dropna=False))  # Includes NaN counts

print("-------------------------------------")
print("")

# Missing values for selected columns
print(df[['customer_age', 'page_views', 'product_category']].isna().sum())

print("-------------------------------------")
print("")

# Statistics grouped by a categorical column
print(df.groupby('product_category')['price'].describe())

print("-------------------------------------")
print("")

import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for numerical columns
sns.boxplot(data=df[['sales_volume', 'price']])
plt.show()

# Bar plot for categorical counts
df['product_category'].value_counts().plot(kind='bar')
plt.show()

print("-------------------------------------")
print("")

plt.scatter(df['sales_volume'], df['price'], color='blue')
plt.show()

print("-------------------------------------")
print("")

# Replace numerical missing values with mean
df['sales_volume'].fillna(df['sales_volume'].mean(), inplace=True)

# Replace numerical missing values with median
df['price'].fillna(df['price'].median(), inplace=True)

df['product_category'].fillna(df['product_category'].mode()[0], inplace=True)

df.dropna(subset=['customer_age'], inplace=True)

print("After numerical imputation:")
print(df.isna().sum())

specific_nans = df[df[['customer_age']].isna().any(axis=1)]
print('specific_nans = ')
print(specific_nans)

print("-------------------------------------")
print("")

import pandas as pd
import numpy as np

# Sample DataFrame
df_with_missing_values = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'age': [25, np.nan, 30, np.nan],
    'income': [50000, 60000, np.nan, np.nan],
    'city': ['NY', 'LA', np.nan, 'Chicago']
})

print("Original DataFrame:")
print(df_with_missing_values)

print("\nRows with ANY missing values:")
print(df_with_missing_values[df_with_missing_values.isna().any(axis=1)])

print("\nRows missing 'age':")
print(df_with_missing_values[df_with_missing_values['age'].isna()])

print("\nComplete rows (no missing values):")
print(df_with_missing_values.dropna())

print("-------------------------------------")
print("")

import pandas as pd
import numpy as np

# Sample DataFrame
df_with_missing_values1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'age': [25, np.nan, 30, np.nan],
    'city': ['NY', 'LA', np.nan, 'Chicago']
})

# Drop rows where 'age' is missing
df_cleaned0 = df_with_missing_values1.dropna(subset=['age'])
print("df_cleaned0 = ")
print(df_cleaned0)

# Drop rows where ANY of the specified columns have missing values
df_cleaned1 = df_with_missing_values1.dropna(subset=['age', 'city'])
print("df_cleaned1 = ")
print(df_cleaned1)

# Drop rows where ALL specified columns have missing values
df_cleaned2 = df_with_missing_values1.dropna(subset=['age', 'city'], how='all')
print("df_cleaned2 = ")
print(df_cleaned2)

print("-------------------------------------")
print("")

def filterOutlierV1(x, col_name, mean_val, std_val, threshold=3.0):
    if np.abs(x[col_name] - mean_val) > threshold * std_val:
        return True
    return False


def removeOutlierV1(df_in, col_name, threshold=1.0):
    mean_val = df_in[col_name].mean()
    std_val = df_in[col_name].std()

    mask=df_in.apply(lambda row: filterOutlierV1(row, col_name, mean_val, std_val, threshold), axis=1)
    print("mask = ")
    print(mask)

    df_in.loc[mask, col_name] = df_in[col_name].median()
    return

def removeOutlierV2(df_in, col_name, threshold=1.0):
    mean_val = df_in[col_name].mean()
    std_val = df_in[col_name].std()

    mask=df_in.apply(lambda row: filterOutlierV1(row, col_name, mean_val, std_val, threshold), axis=1)
    print("mask = ")
    print(mask)

    df_in = df_in.loc[~mask]
    return df_in

print('before remove outliers')
print('df[\'price\'].describe() = ')
print(df['price'].describe())

print(df['price'])

#removeOutlierV1(df, 'price')
df = removeOutlierV2(df, 'price')

print('after remove outliers')
print('df[\'price\'].describe() = ')
print(df['price'].describe())

print(df['price'])


print("-------------------------------------")
print("")

import pandas as pd

# Sample DataFrames
df_join1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_join2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Stack vertically (axis=0)
result0 = pd.concat([df_join1, df_join2], axis=0)
print('result0 = ')
print(result0)

# Stack horizontally (axis=1)
result1 = pd.concat([df_join1, df_join2], axis=1)
print('result1 = ')
print(result1)


print("-------------------------------------")
print("")

# Sample DataFrames with a common key
df1 = pd.DataFrame({'key': ['a', 'b'], 'value': [1, 2]})
df2 = pd.DataFrame({'key': ['b', 'c'], 'value': [3, 4]})

print('df1 = ')
print(df1)
print()

print('df2 = ')
print(df2)
print()

# Inner join (default)
result_inner = pd.merge(df1, df2, on='key')
print('result_inner = ')
print(result_inner)
print()

# Left join
result_left = pd.merge(df1, df2, on='key', how='left')
print('result_left = ')
print(result_left)
print()

# Right join
result_right = pd.merge(df1, df2, on='key', how='right')
print('result_right = ')
print(result_right)
print()

# Outer join
result_outer = pd.merge(df1, df2, on='key', how='outer')
print('result_outer = ')
print(result_outer)
print()

print("-------------------------------------")
print("")

import pandas as pd
import numpy as np

# Sample DataFrame
df = pd.DataFrame({
    'values': [1, 10, 100, 1000]
})

print("before transform, df")
print(df)
print()

# Natural log transform (ln)
df['log_values'] = np.log(df['values'])

print("after transform, df")
print(df)
print()

print("-------------------------------------")
print("")

import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

print("before multiply, df")
print(df)
print()

# Multiply columns 'A' and 'B'
df['product'] = df['A'] * df['B']

print("after multiply, df")
print(df)
print()

print("-------------------------------------")
print("")

from sklearn.model_selection import train_test_split
import pandas as pd

# Sample dataset with class imbalance
data = pd.DataFrame({
    'feature': range(100),
    'label': [0] * 90 + [1] * 10  # 90% class 0, 10% class 1
})

# Stratified split (preserves class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    data['feature'],
    data['label'],
    test_size=0.2,
    stratify=data['label'],  # Key parameter!
    random_state=42
)

# Check class distribution in train/test sets
print("y_Train:", y_train.value_counts(normalize=True))
print("y_Test:", y_test.value_counts(normalize=True))

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(data['feature'], data['label']):
    X_train = data.iloc[train_idx]
    X_test = data.iloc[test_idx]

# Check class distribution in train/test sets
print("X_Train:", X_train.value_counts(normalize=True))
print("X_Test:", X_test.value_counts(normalize=True))

print("-------------------------------------")
print("")



print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")

print("-------------------------------------")
print("")

print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")

print("-------------------------------------")
print("")

print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")


print("-------------------------------------")
print("")
