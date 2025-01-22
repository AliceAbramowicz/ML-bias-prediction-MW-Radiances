import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_dini = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2023.csv")
df_dutch = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2021.csv")

print("Missing values in dini long dataset:", df_dini.isnull().sum())
df_dini = df_dini.drop(['sat', 'sensor', 'predxcov_1'], axis=1)
df_dutch = df_dutch.drop(['sat', 'sensor', 'predxcov_1'], axis=1)

plt.figure(figsize=(6, 6))
sns.heatmap(data=df_dini.corr(), mask=np.triu(df_dini.corr()), annot=False, fmt=".2f", cmap='RdYlBu_r', linewidth=1)
plt.title('Correlation Heatmap: Dini domain all data groups')
plt.tight_layout()
plt.savefig('correlation_dini.png')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(data=df_dutch.corr(), mask=np.triu(df_dutch.corr()), annot=False, fmt=".2f", cmap='RdYlBu_r', linewidth=1)
plt.title('Correlation Heatmap: Dutch domain all data groups')
plt.tight_layout()
plt.savefig('correlation_dutch.png')
plt.show()
