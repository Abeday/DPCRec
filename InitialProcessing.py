import pandas as pd
from tqdm.auto import tqdm
import numpy as np


K = 50  # sequence length

df_train = pd.read_csv('data/nyc/NYC_train.csv')
df_test = pd.read_csv('data/nyc/NYC_test.csv')
df_val = pd.read_csv('data/nyc/NYC_val.csv')

cut_train = df_train[['user_id', 'POI_id', 'latitude', 'longitude', 'UTC_time']]
cut_test = df_test[['user_id', 'POI_id', 'latitude', 'longitude', 'UTC_time']]
cut_val = df_val[['user_id', 'POI_id', 'latitude', 'longitude', 'UTC_time']]

cut_combine = pd.concat([cut_train, cut_test, cut_val], axis=0)
sorted_data = cut_combine.sort_values(by='user_id')
sorted_data.reset_index(drop=True, inplace=True)
sorted_data['user_id'] = sorted_data['user_id'] - 1

# regenerate poi id
value_mapping = {}
current_max = 0
for i in tqdm(range(len(sorted_data))):
    value = sorted_data.loc[i, 'POI_id']
    if value not in value_mapping:
        if i == 0:
            value_mapping[value] = 0
        else:
            value_mapping[value] = current_max + 1
            current_max += 1
    sorted_data.loc[i, 'POI_id'] = value_mapping[value]

# split data
checkin_ori = sorted_data[['user_id', 'POI_id', 'UTC_time']]
coos_ori = sorted_data[['POI_id', 'latitude', 'longitude']]

# get check in top K and hour
value_counts = checkin_ori['user_id'].value_counts()
filtered_values = value_counts[value_counts >= K].index
checkin_K = pd.DataFrame()
for value in filtered_values:  # getting
    subset = checkin_ori[checkin_ori['user_id'] == value]
    if len(subset) > K:
        subset = subset.head(K)
    checkin_K = pd.concat([checkin_K, subset])
checkin_K = checkin_K.sort_values(by='user_id')
checkin_K.reset_index(drop=True, inplace=True)
checkin_final = checkin_K.copy()
checkin_final['UTC_time'] = checkin_K['UTC_time'].str.slice(11, 13)  # get hour

# for coos
coos_unique = coos_ori.drop_duplicates(subset='POI_id')

# get numpy mat
checkin_np = checkin_final.to_numpy(dtype=int)
coos_np = coos_unique.to_numpy(dtype=float)

np.savetxt('data/nyc/checkins.txt', checkin_np)
np.savetxt('data/nyc/poi_coos.txt', coos_np)
