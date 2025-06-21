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