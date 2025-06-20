import os
from utils import *


def return_root_mat(args):

    root_mat_path = 'data/' + args.dataset + '/root_mat.npy'
    checkins_path = 'data/' + args.dataset + '/checkins.txt'
    coos_path = 'data/' + args.dataset + '/poi_coos.txt'
    if os.path.exists(root_mat_path):
        print(f'Step 1 : {args.dataset} data prepare')
        print('--exist the root mat, could run directly', end='\n')
        mat_root = np.load(root_mat_path)
    else:
        print(f'Step 1 : {args.dataset} data prepare')
        print('--need generate the root mat')
        checkin = np.loadtxt(checkins_path)   # contain user_id, poi_id, timestamp
        coos = np.loadtxt(coos_path)   # contain poi_id, latitude, longitude
        coos = coos[coos[:, 0].argsort()]   # sorting by point of poi id

        # checkin[:, 2] = timestamps_to_hours(checkin[:, 2])  # convert timestamp to hour

        intervals_mat = np.zeros((checkin.shape[0], 2))
        uid = None
        hour_prev = None
        lat1 = None
        lon1 = None

        for row in range(checkin.shape[0]):  # rows number
            # the checkin row(user, poi, hour)
            # the coos row (poi, latitude, longitude)

            if uid != checkin[row, 0]:  # record the first check-in and continue
                uid = checkin[row, 0]
                intervals_mat[row, :] = 0
                item_prev = checkin[row, 1]
                lat1 = coos[int(item_prev), 1]
                lon1 = coos[int(item_prev), 2]
                hour_prev = checkin[row, 2]
                continue

            item = checkin[row, 1]
            lat2 = coos[int(item), 1]
            lon2 = coos[int(item), 2]
            hour = checkin[row, 2]

            hour_interval = (hour-hour_prev) if (hour-hour_prev) > 0 else (hour-hour_prev+24)
            # space_interval = haversine(lat2, lon2, lat1, lon1)  # whether to keep decimals
            space_interval = np.floor(haversine(lat2, lon2, lat1, lon1))

            intervals_mat[row, :] = [hour_interval, space_interval]

            lat1 = lat2
            lon1 = lon2
            hour_prev = hour
        checkin[:, -1] = checkin[:, -1] // args.time_size  # convert hour to time box
        mat_root = np.hstack((checkin, intervals_mat))  # row:(user_id, item_id, hour, hour_interval, space_interval)
        np.save(root_mat_path, mat_root)
        print(f'--successfully generate, saved at : data/{args.dataset}/mat_root.npy', end='\n')
    return mat_root


class Seqs_Build(object):

    def __init__(self, root):

        users, pois, time, interval, distance = root[:, 0], root[:, 1], root[:, 2], root[:, 3], root[:, 4]

        # for avoiding the padding 0
        pois = pois + 1
        time = time + 1

        user_unique, indices, counts_all = np.unique(users, return_index=True, return_counts=True)
        poi_unique = np.unique(pois)

        # the next three are for embedding
        self.dis_emb_num = int(max(distance)) + 1  # how many distance
        self.poi_emb_num = int(max(pois)) + 1  # how many pois
        self.user_emb_num = int(max(users)) + 1  # how many users

        self.poi_num = int(max(poi_unique))  # how many pois there are
        self.long_seq_length = max(counts_all)

        self.user = users.reshape((-1, self.long_seq_length))
        self.poi = pois.reshape((-1, self.long_seq_length))
        self.time = time.reshape((-1, self.long_seq_length))
        self.interval = interval.reshape((-1, self.long_seq_length))
        self.distance = distance.reshape((-1, self.long_seq_length))

        self.mask = np.ones((self.long_seq_length, self.long_seq_length, self.long_seq_length))
        for i in range(self.mask.shape[0]):
            self.mask[i, :i + 1, :i + 1] = 0
