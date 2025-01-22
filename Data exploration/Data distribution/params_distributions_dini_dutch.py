import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_dini = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2023.csv")
df_dutch = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2021.csv")

pred_ids=np.sort(df_dutch['pred_id'].unique())

combined_data = pd.concat([df_dini, df_dutch])
min_param = combined_data['param'].min()
max_param = combined_data['param'].max()
binwidth = (max_param - min_param) / 40

pred = {
        0: 'constant',
        1: '1000-300hPa thickness',
        2: '200-50hPa thickness',
        3: 'nadir view angle',
        4: 'nadir view angle **2',
        5: 'nadir view angle **3'
}

fig, axs = plt.subplots(len(pred_ids), figsize=(15, 10))

for j, pred_id in enumerate(pred_ids):
    dini_params = df_dini[df_dini['pred_id'] == pred_id]['param']
    dutch_params = df_dutch[df_dutch['pred_id'] == pred_id]['param']
    sns.histplot(dini_params, kde=False, binwidth=binwidth, color='blue', alpha=0.8, ax=axs[j])
    sns.histplot(dutch_params, kde=False, binwidth=binwidth, color='red', alpha=0.8, ax=axs[j])
        
    axs[j].set_title(f'Predictor {pred[j]}')
    axs[j].set_xlabel('Betas')
    axs[j].set_ylabel('Frequency')
        
    axs[j].set_xlim(min_param, max_param)
    axs[j].set_ylim(0, 3700)
    
    sample_size_dini = len(dini_params)
    sample_size_dutch = len(dutch_params)
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), labels=['DINI domain', 'Dutch domain'])
    axs[j].legend()

plt.tight_layout()
plt.savefig('Betas_Distribution_per_pred.png')
plt.show()


