import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 80000, 60000, 90000],
    'credit_score': [650, 700, 720, 680]
})

# Create interaction terms
df['age_income_interaction'] = df['age'] * df['income']
df['age_credit_interaction'] = df['age'] * df['credit_score']
print("Interaction Features:")
print(df[['age', 'income', 'age_income_interaction']].head())

# Create crossed feature by combining categories
df['location'] = ['NY', 'CA', 'NY', 'TX']
df['gender'] = ['M', 'F', 'F', 'M']
df['location_gender_cross'] = df['location'] + '_' + df['gender']
print("\nFeature Cross Example:")
print(df[['location', 'gender', 'location_gender_cross']].head())

# Create ratio features
df['income_to_age_ratio'] = df['income'] / df['age']
df['credit_to_income_ratio'] = df['credit_score'] / df['income'] * 1000
print("\nRatio Features:")
print(df[['income', 'age', 'income_to_age_ratio']].head())

# Create polynomial features manually
df['age_squared'] = df['age'] ** 2
df['income_squared'] = df['income'] ** 2
print("\nQuadratic Features:")
print(df[['age', 'age_squared', 'income', 'income_squared']].head())

from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features (degree=2 includes squares and interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['age', 'income']])
poly_df = pd.DataFrame(poly_features,
                      columns=['age', 'income', 'age^2', 'age*income', 'income^2'])
print("\nscikit-learn Polynomial Features:")
print(poly_df.head())

# Financial domain example
df['debt_to_income'] = df['income'] * 0.3 - df['credit_score']  # Hypothetical formula
print("\nDomain-Specific Features:")
print(df[['income', 'credit_score', 'debt_to_income']].head())

# For time series data
df['date'] = pd.date_range('2023-01-01', periods=4)
df['days_since'] = (df['date'].max() - df['date']).dt.days
df['income_per_day'] = df['income'] / df['days_since']
print("\nTime-Based Ratios:")
print(df[['date', 'income', 'income_per_day']])

print("df")
print(df)
print("\n")