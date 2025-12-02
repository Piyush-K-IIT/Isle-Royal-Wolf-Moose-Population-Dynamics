import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load data and clean up NA
# -------------------------------
df = pd.read_csv(
    "wolf_moose_2019.csv",
    na_values=["NA yet", "N/A yet", "NA Yet", "N/A Yet", "NA", "N/A"]
)

# Columns to process
target_cols = [
    'July-Sept (temp, F)',
    'Apr-May (temp, F)',
    'Jan-Feb (temp, F)',
    'May-Aug (precip, inches)',
    'snow.depth (cm)'
]

# Convert to numeric; invalid strings → NaN
df[target_cols] = df[target_cols].apply(pd.to_numeric, errors='coerce')

# -------------------------------
# 2. Interpolate missing values
# -------------------------------
for col in target_cols:
    before_missing = df[col].isna().sum()
    df[col] = df[col].interpolate(method='linear', limit_direction='both')
    after_missing = df[col].isna().sum()
    print(f"{col}: filled {before_missing - after_missing} missing values (remaining: {after_missing})")

# -------------------------------
# 3. Plot each variable separately
# -------------------------------
for col in target_cols:
    plt.figure(figsize=(10, 6))
    plt.plot(df['year'], df[col], marker='o', linewidth=2, label=f"{col} (Interpolated)")
    plt.title(f"{col} over Years (Spline Interpolated)")
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4. Optional: Save cleaned data
# -------------------------------
# Save selected columns: identifiers + cleaned climate targets
columns_to_save = ['year', 'moose', 'wolves'] + target_cols
df[columns_to_save].to_csv("cleaned_climate_data.csv", index=False)
print("\nCleaned data saved to 'cleaned_climate_data.csv' with columns:", columns_to_save)