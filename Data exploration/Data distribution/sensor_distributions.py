import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_dini = pd.read_csv("/perm/nld3863/create_dataset_all/varbc_dataset_dini_common_sat_sen.csv")
df_dutch = pd.read_csv("/perm/nld3863/create_dataset_all/varbc_dataset_dutch_common_sat_sen.csv")

# DINI DOMAIN:
sensor_3_data_dini = df_dini[df_dini['sensor'] == 3]['param']
sensor_15_data_dini = df_dini[df_dini['sensor'] == 15]['param']
sensor_73_data_dini = df_dini[df_dini['sensor'] == 73]['param']

min_value = min(sensor_3_data_dini.min(), sensor_15_data_dini.min(), sensor_73_data_dini.min())
max_value = max(sensor_3_data_dini.max(), sensor_15_data_dini.max(), sensor_73_data_dini.min())
print("min value:", min_value)
print("max value:", max_value)
binwidth = 0.1

plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
sns.histplot(sensor_3_data_dini, kde=False, color='purple', binwidth=binwidth)
plt.title('Sensor AMSUA')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

plt.subplot(3, 1, 2)
sns.histplot(sensor_15_data_dini, kde=False, color='darkturquoise', binwidth=binwidth)
plt.title('Sensor MHS')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

plt.subplot(3, 1, 3)
sns.histplot(sensor_73_data_dini, kde=False, color='orange', binwidth=binwidth)
plt.title('Sensor MWHS2')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

#plt.suptitle('Dini Domain: Distribution of Betas per Sensor', fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("DINI_Betas_distribution_per_sensor.png")
plt.show()

# DUTCH DOMAIN:
sensor_3_data_dutch = df_dutch[df_dutch['sensor'] == 3]['param']
sensor_15_data_dutch = df_dutch[df_dutch['sensor'] == 15]['param']
sensor_73_data_dutch = df_dutch[df_dutch['sensor'] == 73]['param']

#min_value = min(sensor_3_data_dutch.min(), sensor_15_data_dutch.min(), sensor_73_data_dutch.min())
#max_value = max(sensor_3_data_dutch.max(), sensor_15_data_dutch.max(), sensor_73_data_dutch.min())

plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
sns.histplot(sensor_3_data_dutch, kde=False, color='purple', binwidth=binwidth)
plt.title('Sensor AMSUA')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

plt.subplot(3, 1, 2)
sns.histplot(sensor_15_data_dutch, kde=False, color='darkturquoise', binwidth=binwidth)
plt.title('Sensor MHS')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

plt.subplot(3, 1, 3)
sns.histplot(sensor_73_data_dutch, kde=False, color='orange', binwidth=binwidth)
plt.title('Sensor MWHS2')
plt.xlabel('Betas')
plt.ylabel('Frequency')
plt.xlim(min_value, max_value)

#plt.suptitle('Dutch Domain: Distribution of Betas per Sensor', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("DUTCH_Betas_distribution_per_sensor.png")
plt.show()


