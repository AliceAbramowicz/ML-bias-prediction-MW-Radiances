import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_dini = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2023.csv")
df_dutch = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2021.csv")

combined_data = pd.concat([df_dini, df_dutch])
min_param = combined_data['param'].min()
max_param = combined_data['param'].max()
binwidth = (max_param - min_param) / 40

dini_params = df_dini['param']
dutch_params = df_dutch['param']
sns.histplot(dini_params, kde=False, binwidth=binwidth, color='blue')
sns.histplot(dutch_params, kde=False, binwidth=binwidth, color='red')
plt.title(f'Histogram: bias parameters')
plt.xlabel('Betas', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(min_param, max_param)
plt.ylim(0, None)
plt.tight_layout()
plt.legend(['DINI domain', 'Dutch domain'], loc='upper right', bbox_to_anchor=(1, 1))
plt.savefig('Betas_Distribution_per_domain.png')
plt.show()
