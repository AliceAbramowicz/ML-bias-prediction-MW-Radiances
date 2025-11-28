import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_dini = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/varbc_DINI_common_sat_sen.csv")
df_dutch = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/varbc_dutch_common_sat_sen.csv")

# DINI DOMAIN:
sensor_3_data_dini = df_dini[df_dini['sensor'] == 3]['param']
sensor_15_data_dini = df_dini[df_dini['sensor'] == 15]['param']
sensor_73_data_dini = df_dini[df_dini['sensor'] == 73]['param']

# DUTCH DOMAIN:
sensor_3_data_dutch = df_dutch[df_dutch['sensor'] == 3]['param']
sensor_15_data_dutch = df_dutch[df_dutch['sensor'] == 15]['param']
sensor_73_data_dutch = df_dutch[df_dutch['sensor'] == 73]['param']

min_value = min(sensor_3_data_dini.min(), sensor_15_data_dini.min(), sensor_73_data_dini.min(),
                sensor_3_data_dutch.min(), sensor_15_data_dutch.min(), sensor_73_data_dutch.min())
max_value = max(sensor_3_data_dini.max(), sensor_15_data_dini.max(), sensor_73_data_dini.max(),
                sensor_3_data_dutch.max(), sensor_15_data_dutch.max(), sensor_73_data_dutch.max())
binwidth = 0.1

plt.figure(figsize=(6, 6))

# AMSU-A plot
plt.subplot(3, 1, 1)
sns.histplot(sensor_3_data_dini, kde=False, color='blue', binwidth=binwidth, label='Dini')
sns.histplot(sensor_3_data_dutch, kde=False, color='red', binwidth=binwidth, label='Dutch')
plt.title('AMSU-A')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)
plt.legend()

# MHS plot
plt.subplot(3, 1, 2)
sns.histplot(sensor_15_data_dini, kde=False, color='blue', binwidth=binwidth, label='Dini')
sns.histplot(sensor_15_data_dutch, kde=False, color='red', binwidth=binwidth, label='Dutch')
plt.title('MHS')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

# MWHS2 plot
plt.subplot(3, 1, 3)
sns.histplot(sensor_73_data_dini, kde=False, color='blue', binwidth=binwidth, label='Dini')
sns.histplot(sensor_73_data_dutch, kde=False, color='red', binwidth=binwidth, label='Dutch')
plt.title('MWHS2')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Betas_distribution_per_sensor.png")
plt.show()

