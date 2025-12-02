import pandas as pd

# Load the DataFrame from the file path you provided
df = pd.read_csv("cleaned_climate_data.csv")

# Calculate Pearson Correlation (Linear)
pearson_corr = df.corr(method='pearson').round(3)

# Calculate Spearman Correlation (Monotonic)
spearman_corr = df.corr(method='spearman').round(3)

print("--- Pearson Correlation Matrix (r) ---")
# Replaced .to_markdown() with the standard .to_string()
print(pearson_corr.to_string())

print("\n--- Spearman Correlation Matrix (rho) ---")
# Replaced .to_markdown() with the standard .to_string()
print(spearman_corr.to_string())