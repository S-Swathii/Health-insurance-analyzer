import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('health insurance.csv')

# Define numerical features
numerical_features = ['age', 'bmi', 'children', 'charges']

# Preprocessing
# Standardize numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Perform PCA
pca = PCA(n_components=None)  # Use all components
principal_components = pca.fit_transform(df[numerical_features])

# Create DataFrame for loadings
loadings_df = pd.DataFrame(pca.components_, columns=numerical_features)

# Print loadings
print("PCA Loadings:")
print(loadings_df)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print explained variance ratio for each component
print('\nExplained variance ratio for each component:')
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"Principal Component {i}: {ratio:.4f}")

# PCA Analysis Inference
print("\nPCA Analysis Inference:")
print("PCA effectively reduces the dimensionality of the dataset,with the first principal component capturing the largest variance, primarily influenced by charges and age.")
print("The second principal component is dominated by children, indicating a separate dimension of variability.")
print("Together, the first two components explain a significant portion of the variance, suggesting that charges, age, and children are key variables in the dataset.")

# Perform Factor Analysis
n_factors = len(numerical_features)  # Specify the number of factors
fa = FactorAnalysis(n_components=n_factors, random_state=0)
factor_components = fa.fit_transform(df[numerical_features])

# Create DataFrame for factor loadings
factor_loadings_df = pd.DataFrame(fa.components_, columns=numerical_features)

# Print factor loadings
print("\nFactor Analysis Loadings:")
print(factor_loadings_df)

# Explained variance
# Note: Factor Analysis doesn't provide explained variance ratio directly like PCA.
# However, we can infer it from the loadings.
explained_variance = factor_loadings_df.var(axis=1)

# Print explained variance for each factor
print('\nExplained variance for each factor:')
for i, var in enumerate(explained_variance, 1):
    print(f"Factor {i}: {var:.4f}")

# Factor Analysis Inference
print("\nFactor Analysis Inference:")
print("Factor Analysis reveals that most of the variance is captured by a single factor, heavily influenced by charges.")
print("The subsequent factors do not contribute significantly to the explained variance, indicating that a single underlying factor (likely related to insurance charges) is predominant.")

# Comparison
print("\nComparison:")
print("Both PCA and Factor Analysis identify charges as a major contributing factor.")
print("PCA provides a clearer picture of the individual contributions of age, bmi, and children through the different principal components.")
print("Factor Analysis simplifies the structure to one dominant factor, which might be useful for identifying a single underlying trait in the dataset.")
