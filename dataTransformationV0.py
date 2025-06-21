##from sklearn.preprocessing import StandardScaler
##import numpy as np
##import pandas as pd
##
##data = np.array([[1, 2], [3, 4], [5, 6]])
##
##scaler = StandardScaler()
##scaled_data = scaler.fit_transform(data)
##print("Standardized data:\n", scaled_data)
##print("Mean:", scaler.mean_, "Std:", scaler.scale_)
##
##from sklearn.preprocessing import MinMaxScaler
##
##scaler = MinMaxScaler(feature_range=(0, 1))
##minmax_data = scaler.fit_transform(data)
##print("\nMin-Max scaled data:\n", minmax_data)
##print("Data min:", scaler.data_min_, "Data max:", scaler.data_max_)
##
##from sklearn.preprocessing import RobustScaler
##
##robust_scaler = RobustScaler()
##robust_data = robust_scaler.fit_transform(data)
##print("\nRobust scaled data:\n", robust_data)
##print("Median:", robust_scaler.center_, "IQR:", robust_scaler.scale_)
##
##data = np.array([1, 10, 100, 1000])
##
### Adding 1 to avoid log(0)
##log_data = np.log1p(data)
##print("\nLog transformed data:\n", log_data)
##
##from sklearn.preprocessing import KBinsDiscretizer
##
##est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
##binned_data = est.fit_transform(data.reshape(-1, 1))
##print("\nEqual-width bins:\n", binned_data.flatten())
##
##est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
##quantile_data = est.fit_transform(data.reshape(-1, 1))
##print("\nQuantile bins:\n", quantile_data.flatten())
##
##def clip_outliers(series, lower=0.05, upper=0.95):
##    lower_bound = series.quantile(lower)
##    upper_bound = series.quantile(upper)
##    return series.clip(lower_bound, upper_bound)
##
##data = pd.Series([1, 2, 3, 4, 5, 100])
##clipped_data = clip_outliers(data)
##print("\nClipped data:\n", clipped_data)
##
##data = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])
##
##robust_scaler = RobustScaler()
##robust_data = robust_scaler.fit_transform(data)
##print("\nRobust scaled data with outliers:\n", robust_data)
##
##from sklearn.preprocessing import PolynomialFeatures
##
##poly = PolynomialFeatures(degree=2, include_bias=False)
##poly_data = poly.fit_transform(data)
##print("\nPolynomial features:\n", poly_data)
##
##from sklearn.preprocessing import PolynomialFeatures
##
##interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
##interaction_data = interaction.fit_transform(data)
##print("\nInteraction terms:\n", interaction_data)