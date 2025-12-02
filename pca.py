import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


df = pd.read_csv("cleaned_climate_data.csv")

# The variable containing the names of the columns i want to use 
climate = ["July-Sept (temp, F)", "Apr-May (temp, F)", "Jan-Feb (temp, F)", "snow.depth (cm)"]

# --- PCA EXECUTION START ---

# 2. Isolate Data
# Get the values of the target columns as a NumPy array
X = df[climate].values

print(f"--- Running PCA on {len(climate)} features and {len(df)} samples ---")

# 3. Standardize the Data (MANDATORY STEP)
# Each feature must have mean=0 and standard deviation=1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Perform PCA
# We choose to keep all components (n_components=len(climate)) initially 
# to calculate a complete Loading Matrix. You can change this to a specific number (e.g., 2).
n_components = len(climate)
pca = PCA(n_components=n_components)

# Fit and transform the scaled data
principal_components = pca.fit_transform(X_scaled)

# 5. Create the Loading Matrix
# The Loading Matrix is stored in pca.components_. 
# It represents the eigenvectors (principal axes) in terms of the original features.
# The values (loadings) indicate the correlation between the original variables and the new components.
loadings = pd.DataFrame(
    pca.components_.T,  # Transpose to get features as rows
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=climate
)

# 6. Optional: Create the Principal Component Scores DataFrame
# This is the transformed data (the result of the projection)
pca_scores_df = pd.DataFrame(
    data=principal_components,
    columns=[f'PC{i+1}' for i in range(n_components)]
)

# 7. Print Results

print("\n\n--- ⚠️ Loading Matrix (The Core Result) ⚠️ ---")
print("These are the weights (loadings) of the original features for each Principal Component (PC).")
print("Higher absolute values indicate a stronger influence on that specific PC.")
print(loadings)

print("\n\n--- Explained Variance Ratio ---")
print("This tells you how much information (variance) each component captured.")
explained_variance_df = pd.DataFrame({
    'Variance Explained': pca.explained_variance_ratio_,
    'Cumulative Variance': pca.explained_variance_ratio_.cumsum()
}, index=[f'PC{i+1}' for i in range(n_components)])
print(explained_variance_df)

print("\n\n--- Transformed Data (PCA Scores) ---")
print("Your original data projected onto the new principal axes.")
print(pca_scores_df.head())