##import pandas as pd
##from sklearn.preprocessing import OneHotEncoder
##
### Sample data
##data = pd.DataFrame({
##    'color': ['red', 'blue', 'green', 'blue', 'red'],
##    'size': ['S', 'M', 'L', 'M', 'XL']
##})
##
### One-hot encode using pandas
##onehot_pd = pd.get_dummies(data, columns=['color', 'size'])
##print("Pandas one-hot:\n", onehot_pd.head())
##
### Using scikit-learn
##encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first avoids multicollinearity
##onehot_sk = encoder.fit_transform(data[['color', 'size']])
##print("\nScikit-learn one-hot:\n", onehot_sk[:3])
##
##from sklearn.preprocessing import LabelEncoder
##
### For ordinal data with meaningful order
##size_order = ['XS', 'S', 'M', 'L', 'XL']
##data['size_encoded'] = data['size'].apply(lambda x: size_order.index(x))
##print("\nManual label encoding:\n", data[['size', 'size_encoded']].head())
##
### Using scikit-learn (caution: doesn't preserve order)
##le = LabelEncoder()
##data['color_encoded'] = le.fit_transform(data['color'])
##print("\nLabelEncoder output:\n", data[['color', 'color_encoded']].head())
##
##from category_encoders import TargetEncoder
##from sklearn.model_selection import train_test_split
##
### Create target variable
##data['target'] = [1, 0, 1, 1, 0]
##
### Target encoding with regularization
##X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)
##encoder = TargetEncoder(cols=['color'])
##encoder.fit(X_train['color'], X_train['target'])
##data['color_encoded'] = encoder.transform(data['color'])
##print("\nTarget encoding:\n", data[['color', 'color_encoded']].head())
##
### Count frequency of each category
##freq_map = data['color'].value_counts(normalize=True)
##data['color_freq'] = data['color'].map(freq_map)
##print("\nFrequency encoding:\n", data[['color', 'color_freq']].head())
##
##from category_encoders import BinaryEncoder
##
##encoder = BinaryEncoder(cols=['color'])
##binary_encoded = encoder.fit_transform(data['color'])
##print("\nBinary encoding:\n", binary_encoded.head())
##
##from sklearn.feature_extraction import FeatureHasher
##
##hasher = FeatureHasher(n_features=4, input_type='string')
##hashed = hasher.transform(data['color'].apply(lambda x: [x])).toarray()
##hashed_df = pd.DataFrame(hashed, columns=[f'color_hash_{i}' for i in range(4)])
##print("\nFeature hashing:\n", hashed_df.head())
##
##from tensorflow.keras.layers import Embedding, Input
##from tensorflow.keras.models import Model
##import numpy as np
##
### Prepare data
##categories = data['color'].unique()
##cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
##data['color_idx'] = data['color'].map(cat_to_idx)
##
### Create embedding model
##input_layer = Input(shape=(1,))
##embedding = Embedding(input_dim=len(categories), output_dim=2)(input_layer)
##model = Model(inputs=input_layer, outputs=embedding)
##
### Get embeddings
##embeddings = model.predict(np.array(data['color_idx']))
##print("\nEmbedding vectors:\n", embeddings[:3])
##
### Group rare categories into 'other'
##threshold = 0.1  # Minimum frequency to keep
##freq = data['color'].value_counts(normalize=True)
##data['color_clean'] = data['color'].where(data['color'].isin(freq[freq > threshold].index), 'other')
##print("\nHandling rare categories:\n", data['color_clean'].value_counts())
##
##from sklearn.compose import ColumnTransformer
##from sklearn.pipeline import Pipeline
##from sklearn.ensemble import RandomForestClassifier
##
### Define preprocessing
##preprocessor = ColumnTransformer(
##    transformers=[
##        ('onehot', OneHotEncoder(), ['color']),
##        ('target', TargetEncoder(), ['size'])
##    ],
##    remainder='passthrough'
##)
##
### Create pipeline
##pipeline = Pipeline([
##    ('preprocessor', preprocessor),
##    ('classifier', RandomForestClassifier())
##])
##
### Example usage (X_train, y_train would be your actual data)
### pipeline.fit(X_train, y_train)
##
##from sklearn.model_selection import cross_val_score
##
### Compare encodings
##encoders = {
##    'OneHot': OneHotEncoder(),
##    'Target': TargetEncoder(),
##    'Binary': BinaryEncoder()
##}
##
##for name, encoder in encoders.items():
##    preprocessor = ColumnTransformer(
##        [('encoder', encoder, ['color', 'size'])],
##        remainder='drop'
##    )
##    X_encoded = preprocessor.fit_transform(data, data['target'])
##    scores = cross_val_score(RandomForestClassifier(), X_encoded, data['target'], cv=3)
##    print(f"{name} encoding accuracy: {scores.mean():.3f}")