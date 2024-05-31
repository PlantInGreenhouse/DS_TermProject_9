import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Data Load
file_path = 'termproject/ds_salaries.csv'
data = pd.read_csv(file_path)

# Selected features
features = ['experience_level', 'employment_type', 'job_title', 'remote_ratio', 'company_size']
target = 'salary_in_usd'

# Save the original data before outlier removal
original_data = data.copy()

# Remove outliers in 'salary_in_usd' using the z-score method
z_scores = np.abs((data[target] - data[target].mean()) / data[target].std())
data_cleaned = data[z_scores < 3]

# Data with outliers
outliers = original_data[z_scores >= 3]

# Retain only the selected features in the data
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

# Combine the preprocessed data (using sparse matrices)
preprocessed_data = hstack([
    experience_level_encoded,
    remote_ratio_encoded,
    company_size_encoded,
    employment_type_encoded,
    job_title_encoded,
    target_encoded
])

# Generate column names
feature_names = ['experience_level', 'remote_ratio', 'company_size']
feature_names += list(onehot_encoder_employment.get_feature_names_out(['employment_type']))
feature_names += list(onehot_encoder_job.get_feature_names_out(['job_title']))
feature_names.append(target)

# Convert to DataFrame (convert to dense matrix for saving)
preprocessed_data_df = pd.DataFrame(preprocessed_data.toarray(), columns=feature_names)

# Split the data using the Holdout method
X = preprocessed_data_df.drop(columns=[target])
y = preprocessed_data_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_splits = [(train_index, test_index) for train_index, test_index in kf.split(X)]

# Save the preprocessed data and outliers to CSV files
preprocessed_data_df.to_csv('termproject/preprocessed_data_normalized.csv', index=False)

# Output results
print("Preprocessed Dataset:")
print(preprocessed_data_df.head())
print("\nTraining Set Size:", X_train.shape)
print("Test Set Size:", X_test.shape)
print("Number of 10-Fold Cross-Validation Splits:", len(cv_splits))