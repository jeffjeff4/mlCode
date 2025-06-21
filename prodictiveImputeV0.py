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
