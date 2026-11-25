import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_dini = pd.read_csv("//perm/nld3863/ML-bias-prediction-MW-Radiances/datasets/final_df/big_df_stats_dini_all_data.csv")
df_dutch = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiances/datasets/final_df/big_df_stats_dutch_all_data.csv")

sensors=np.sort(df_dini['sensor'].unique())

combined_data = pd.concat([df_dini, df_dutch])
min_param = combined_data['param'].min()
max_param = combined_data['param'].max()
# binwidth = (max_param - min_param) / 50
# print('binwidth', binwidth)

sensors = {
    3: 'AMSUA',
    15: 'MHS',
    19: 'ATMS',
    73: 'MWHS2',
}

fig, axs = plt.subplots(len(sensors), figsize=(8, 20))

for j, sensor in enumerate(sensors):
    dini_params = df_dini[df_dini['sensor'] == sensor]['param']
    dutch_params = df_dutch[df_dutch['sensor'] == sensor]['param']
    sns.histplot(dini_params, stat="probability", kde=False, binwidth=0.05, binrange=(min_param, max_param), color='blue', alpha=0.6, ax=axs[j])
    sns.histplot(dutch_params, stat="probability", kde=False, binwidth=0.05, binrange=(min_param, max_param), color='red', alpha=0.6, ax=axs[j])
    label = sensors[sensor]
    sample_size_dini = len(dini_params)
    sample_size_dutch = len(dutch_params)
    axs[j].set_title(f'{label} (n_dini={sample_size_dini}, n_dutch={sample_size_dutch})')
    axs[j].set_xlabel('Betas')
    axs[j].set_ylabel('Proportion')
        
    axs[j].set_xlim(min_param, max_param)
    # axs[j].set_ylim(0, 1)

    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), labels=['Dini domain', 'Dutch domain'])
    axs[j].legend()

plt.tight_layout()
plt.savefig('figures/Betas_Distribution_per_sensor.png')
plt.show()