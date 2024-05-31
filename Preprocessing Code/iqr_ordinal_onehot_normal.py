import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Load data
file_path = 'termproject/ds_salaries.csv'
data = pd.read_csv(file_path)

# Selected features
features = ['experience_level', 'employment_type', 'job_title', 'remote_ratio', 'company_size']
target = 'salary_in_usd'

# Save data before outlier removal
original_data = data.copy()

# Remove outliers using the IQR method for 'salary_in_usd'
Q1 = data[target].quantile(0.25)
Q3 = data[target].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_cleaned = data[(data[target] >= lower_bound) & (data[target] <= upper_bound)]

# Outliers data
outliers = original_data[(original_data[target] < lower_bound) | (original_data[target] > upper_bound)]

# Keep only the selected features in the dataset
data_cleaned = data_cleaned[features + [target]]

# Initialize encoders and scaler
ordinal_encoder = OrdinalEncoder()
onehot_encoder_employment = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
onehot_encoder_job = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
normalizer = Normalizer()

# Encode ordered categorical features
data_cleaned[['experience_level', 'company_size']] = ordinal_encoder.fit_transform(data_cleaned[['experience_level', 'company_size']])

# One-hot encode unordered categorical features
employment_type_encoded = onehot_encoder_employment.fit_transform(data_cleaned[['employment_type']])
job_title_encoded = onehot_encoder_job.fit_transform(data_cleaned[['job_title']])

# Normalize numerical features
data_cleaned['remote_ratio'] = normalizer.fit_transform(data_cleaned[['remote_ratio']])

# Convert ordered categorical and numerical features to sparse matrices
experience_level_encoded = csr_matrix(data_cleaned[['experience_level']])
remote_ratio_encoded = csr_matrix(data_cleaned[['remote_ratio']])
company_size_encoded = csr_matrix(data_cleaned[['company_size']])

# Convert the target variable to a sparse matrix
target_encoded = csr_matrix(data_cleaned[[target]])

# Combine preprocessed data into a single sparse matrix
preprocessed_data = hstack([
    experience_level_encoded,
    remote_ratio_encoded,
    company_size_encoded,
    employment_type_encoded,
    job_title_encoded,
    target_encoded
])

# Create feature names
feature_names = ['experience_level', 'remote_ratio', 'company_size']
feature_names += list(onehot_encoder_employment.get_feature_names_out(['employment_type']))
feature_names += list(onehot_encoder_job.get_feature_names_out(['job_title']))
feature_names.append(target)

# Convert the sparse matrix to a dense matrix and then to a DataFrame
preprocessed_data_df = pd.DataFrame(preprocessed_data.toarray(), columns=feature_names)

# Split data using the holdout method
X = preprocessed_data_df.drop(columns=[target])
y = preprocessed_data_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_splits = [(train_index, test_index) for train_index, test_index in kf.split(X)]

# Save the preprocessed data and outliers to CSV files
preprocessed_data_df.to_csv('termproject/preprocessed_data_normalized_iqr.csv', index=False)

# Print results
print("Preprocessed dataset:")
print(preprocessed_data_df.head())
print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Number of 10-Fold cross-validation splits:", len(cv_splits))