import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_dini = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiances/datasets/final_df/big_df_stats_dini_all_data.csv")
df_dutch = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiancesdatasets/final_df/big_df_stats_dutch_all_data.csv")

pred_ids=np.sort(df_dini['pred_id'].unique())

for pred_id in pred_ids:
    print(f"{df_dini.loc[df_dini['pred_id'] == pred_id, 'param'].isna().sum()} NaNs in dini pred {pred_id}")
    d = df_dini.loc[df_dini['pred_id'] == pred_id, 'param'].dropna()
    n = df_dutch.loc[df_dutch['pred_id'] == pred_id, 'param'].dropna()
    # print(pred_id, len(d), len(n), d.min() if len(d) else None, d.max() if len(d) else None)

combined_data = pd.concat([df_dini, df_dutch])
min_param = combined_data['param'].min()
max_param = combined_data['param'].max()
# binwidth = (max_param - min_param) / 50
# print('binwidth', binwidth)

pred_map = {1 : '1 (constant)         ',
            2 : '1000-300hPa thickness',
            3 : '200-50hPa thickness  ',
            4 : 'T_skin               ',
            5 : 'total column water   ',
            6 : '10-2hPa thickness    ',
            7 : '50-5hPa thickness    ',
            8 : 'surface wind speed   ',
            9 : 'nadir view angle  ',
            10 : 'nadir view angle **2 ',
            11 : 'nadir view angle **3 ',
            12 : 'nadir view angle **4 ',
            13 : 'cos solar zen angle  ',
            14 : 'solar elevation      ',
            15 : 'TMI diurnal bias     ',
            16 : 'land or sea ice mask ',
            17 : 'view angle (land)    ',
            18 : 'view angle **2 (land)',
            19 : 'view angle **3 (land)',
            20 : 'ln(rain rate+1)   (1)',
            21 : 'ln(rain rate+1)**2(1)',
            22 : 'ln(rain rate+1)**3(1)',
            23 : 'ln(rain rate+1)   (2)',
            24 : 'ln(rain rate+1)**2(2)',
            25 : 'ln(rain rate+1)**3(2)',
            26 : 'ascent rate  (hPa/s) ',
            27 : 'descent rate (hPa/s) ',
            28 : 'land mask times winds',
            29 : 'day/night            ',
            30 : 'thermal contrast     ',
            31 : 'Radiosonde T 100-850 ',
            32 : 'Radiosonde T  30-200 ',
            33 : 'Radiosonde T   0- 60 ',
            34 : 'Radiosonde T s.elv**1',
            35 : 'Radiosonde T s.elv**2',
            36 : 'Radiosonde log press ',
            37 : 'cos solar zen (full) '
        }
print(type(pred_map.keys))
fig, axs = plt.subplots(len(pred_ids), figsize=(8, 20))

for j, pred_id in enumerate(pred_ids):
    dini_params = df_dini[df_dini['pred_id'] == pred_id]['param']
    dutch_params = df_dutch[df_dutch['pred_id'] == pred_id]['param']
    sns.histplot(dini_params, stat="probability", kde=False, binwidth=0.05, binrange=(min_param, max_param), color='blue', alpha=0.6, ax=axs[j])
    sns.histplot(dutch_params, stat="probability", kde=False, binwidth=0.05, binrange=(min_param, max_param), color='red', alpha=0.6, ax=axs[j])
    label = pred_map[pred_id + 1].strip()
    axs[j].set_title(f'pred {pred_id} {label} (n_dini={len(dini_params)}, n_dutch={len(dutch_params)})')
    axs[j].set_xlabel('Betas')
    axs[j].set_ylabel('Proportion')
        
    axs[j].set_xlim(min_param, max_param)
    # axs[j].set_ylim(0, 1)
    
    sample_size_dini = len(dini_params)
    sample_size_dutch = len(dutch_params)
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), labels=['Dini domain', 'Dutch domain'])
    axs[j].legend()

plt.tight_layout()
plt.savefig('figures/Betas_Distribution_per_pred.png')
plt.show()