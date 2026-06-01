import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
data = pd.read_csv("/csv_v0.txt")
print("data = ")
print(data)
print('\n')

##age,income,gender,purchase_history,target
##25,50000,Male,3,1
##30,80000,Female,5,1
##22,30000,Male,1,0
##35,120000,Female,7,1
##28,45000,Male,2,0
##40,95000,Female,8,1
##19,20000,Male,0,0
##33,110000,Female,6,1
##26,40000,Male,1,0
##45,150000,Female,10,1
##19,20000,Male,0,0
##28,45000,Male,2,0
##26,40000,Male,1,0

# 2. Clean data
data.drop_duplicates(inplace=True)

print('data.drop_duplicates')
print(data)
print('\n')

for column in data.columns:
    print("data[column].dtype = ", data[column].dtype)
    if data[column].dtype in ['number', 'int8', 'int64', 'float32', 'float16']:
        data.fillna(data[column].median(), inplace=True)

print('data after repalcing na with median')
print(data)
print('\n')

# 3. Feature engineering
#data["new_feature"] = data["feature1"] / data["feature2"]

# 4. Encode categorical variables
data = pd.get_dummies(data, columns=["purchase_history"])

#-------------------------------------------------
# column operation
#-------------------------------------------------

print("#-------------------------------------------------")
print('column operation')
print("#-------------------------------------------------")

import pandas as pd

# Sample DataFrame
#df = pd.DataFrame({
#    'A': [1, 2, 3],
#    'B': ['x', 'y', 'z'],
#    'C': [True, False, True]
#})

df = pd.read_csv("/Users/shizhefu0/Desktop/ml/data/deepseek_csv_v0.txt")

print('df original')
print(df)
print('\n')

# Loop through columns
for column in df.columns:
    print(f"Column: {column}")
    print(f"Data type: {df[column].dtype}")
    print(f"Values:\n{df[column]}\n")

# Example: Check if each column has missing values
print("df.apply(lambda col: col.isnull().sum())")
print(df.apply(lambda col: col.isnull().sum()))
print('\n')

# Get the first column (index 0)
first_column = df.iloc[:, 0]
print("first_column")
print(first_column)
print('\n')

print("df Mean (if numeric)")
for column_name, column_data in df.items():
    print(f"Column: {column_name}")
    print(f"Mean (if numeric): {column_data.mean() if pd.api.types.is_numeric_dtype(column_data) else 'N/A'}")
print('\n')

# Convert all strings to uppercase
for column in df.columns:
    if pd.api.types.is_string_dtype(df[column]):
        df[column] = df[column].str.upper()
print("Convert all strings to uppercase")
print(df)
print('\n')


# Example: Standardize all numeric columns
df_numeric = df.select_dtypes(include='number')
df[df_numeric.columns] = (df_numeric - df_numeric.mean()) / df_numeric.std()
print("Standardize all numeric columns")
print(df)
print('\n')

from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_dtype

# Check if a column is numeric
print("is_numeric_dtype(df[\'age\'])")
print(is_numeric_dtype(df['age']))  # True
print('\n')

# Check if a column is string (object)
print("is_string_dtype(df[\'gender\'])")
print(is_string_dtype(df['gender']))   # True
print('\n')

# Check if a column is datetime
print("is_datetime64_dtype(df[\'target\'])")
print(is_datetime64_dtype(df['target']))  # True
print('\n')

# Convert 'A' to float
df['age'] = df['age'].astype('float64')
print("df[\'age\'].dtype")
print(df['age'].dtype)  # Output: float64
print('\n')

# Convert 'B' to categorical
df['gender'] = df['gender'].astype('category')
print("df[\'gender\'].dtype")
print(df['gender'].dtype)  # Output: category
print('\n')
