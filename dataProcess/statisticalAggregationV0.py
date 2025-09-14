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