import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z'],
    'C': [True, False, True]
})

print('df')
print(df)
print('\n')

# Drop single column (e.g., 'B')
df_dropped = df.drop('B', axis=1)
print('df_dropped')
print(df_dropped)
print('\n')

# Drop the second row (index=1)
df_dropped = df.drop(index=df.index[1])
print('df_dropped, 111')
print(df_dropped)
print('\n')

# Modify the DataFrame directly (no copy)
df.drop(['A', 'C'], axis=1, inplace=True)
print('df after dropped')
print(df)
print('\n')


# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5, 6],
    'B': ['x', 'y', 'z', 'a', 'b', 'c'],
    'C': [True, False, True, False, True, False]
})

# Drop rows 'row1' and 'row3'
df_dropped = df.drop(index=df.index[[1, 2]])
print(df_dropped)

print('df_dropped, 222')
print(df_dropped)
print('\n')


# Drop rows where column 'A' > 1
df_dropped = df[df['A'] <= 1]
print('df_dropped, 333')
print(df_dropped)
print('\n')


# Drop rows where column 'A' > 1
df_dropped = df.drop(df[df['A'] >= 6].index)
print('df_dropped, 444')
print(df_dropped)
print('\n')
