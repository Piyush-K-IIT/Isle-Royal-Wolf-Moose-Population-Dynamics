import pandas as pd
import seaborn as sns

df = pd.read_csv("cleaned_climate_data.csv")

corr = df.corr()
print(corr["snow.depth (cm)"].sort_values(ascending=False))
print(corr["moose"].sort_values(ascending=False))
print(corr["wolves"].sort_values(ascending=False))

sns.pairplot(df, vars=["snow.depth (cm)", "Jan-Feb (temp, F)", "Apr-May (temp, F)", "July-Sept (temp, F)", "May-Aug (precip, inches)"])
import matplotlib.pyplot as plt
plt.show()
