import argparse
import collections
import csv
import json
import os
import pickle
import random
import time
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import yaml
from geomloss import SamplesLoss
from matplotlib import pyplot as plt


def load_data(path, isPlot=False):
    dirs = sorted(os.listdir(path))
    if isPlot:
        fig, ax1 = plt.subplots()
        b1, b2 = -6, 6
        ax1.set_xlim(b1, b2)
        ax1.set_ylim(b1, b2)

    input_pos = []
    output_pos = []
    duration = []
    classes = []
    pos_arrs = []
    timestamps_arr = []
    wind_info = []
    bounce = []
    sim_indices = []

    raw_output_pos = []
    color_map = {-1: 'green', 0: 'blue', 1: 'brown', 2: 'red'}
    idx_sim = 0

    round_precision = 4

    x_x_t = []
    for dir in dirs:
        if 'ball' in dir:
            dir_path = os.path.join(path, dir)
            for ball in sorted(os.listdir(dir_path)):
                ball_path = os.path.join(dir_path, ball)
                meta_data_path = os.path.join(ball_path, 'meta_data.json')

                with open(meta_data_path, 'r') as f:
                    meta_data = json.load(f)

                # change the class ID from -1 to 3.
                if meta_data['category_final_pos'] == -1:
                    meta_data['category_final_pos'] = 3

                # get metadata and save to list.
                input_pos.append(meta_data['start_pos'])
                wind_info.append \
                    (meta_data['wind_magnitude'][1: ] +[meta_data['wind_magnitude'][0] -meta_data['start_t']])
                output_pos.append([round(meta_data['final_pos'][0], round_precision),
                                   round(meta_data['final_pos'][1], round_precision),
                                   round(meta_data['final_pos'][2], round_precision),
                                   meta_data['collision_num'],
                                   ])  # (x, y, z, # of collision)
                bounce.append(meta_data['collision_num'])
                raw_output_pos.append(meta_data['final_pos'])
                duration.append(meta_data['duration'])

                classes.append(meta_data['category_final_pos'])

                # read the time series of position x, position z, and the timestamps
                pos_x = []
                pos_z = []
                single_sim_pos = []
                single_sim_timestamp = []
                with open(os.path.join(ball_path, 'position.csv')) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        pos_x.append(round(float(row[1]), round_precision))
                        pos_z.append(round(float(row[3]), round_precision))
                        single_sim_pos.append([round(float(row[1]), round_precision),
                                               round(float(row[2]), round_precision),
                                               round(float(row[3]), round_precision)])
                        single_sim_timestamp.append(int(row[0]))
                pos_arrs.append(single_sim_pos)
                timestamps_arr.append(single_sim_timestamp)

                # save sim idx to list and increment simulation index
                sim_indices.append(idx_sim)
                idx_sim += 1

                x_x_t.append([meta_data['start_pos'][0], meta_data['final_pos'][0], meta_data['duration']])
                if isPlot:
                    ax1.scatter(-meta_data['start_pos'][0], -meta_data['start_pos'][2],
                                color=color_map[meta_data['category_final_pos']])

    if isPlot:
        # plt.tick_params(axis='both', which='major', labelsize=18)
        # plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight',pad_inches = 0.05)
        plt.show()

    input_pos = np.array(input_pos)  # array of input positions
    output_pos = np.array(output_pos)  # array of output positions
    duration = np.array(duration)  # [int], array of the total duration of a simulation
    classes = np.array(classes)  # [int], array of land classes of a simulation
    pos_arrs = np.array(pos_arrs, dtype=object)
    timestamps_arr = np.array(timestamps_arr, dtype=object)   # [[]], where each element is a list of timestamps
    wind_info = np.array(wind_info)     # [[]], where each element is the wind info of a simulation
    sim_indices = np.array(sim_indices)
    raw_output_pos = np.array(raw_output_pos)   # [[]], where each element is the raw output pos without rounding.

    out = (input_pos, output_pos, duration, classes, pos_arrs, timestamps_arr, wind_info, raw_output_pos)
    return out, sim_indices


if __name__ == '__main__':

    random.seed(0)

    data_dir_list = ['output/run_V3.1_b75f25h1015w0.5-0.50.5',
                     ]
    for i in range(34):
        path = 'output/run_V3.1_b75f25h1015w0.5-0.50.5_s{}'.format(i)
        data_dir_list.append(path)

    out_6000_path = 'run_V3.1_b75f25h1015w0.5-0.50.5_balanced_6000.pkl'
    out_full_path = 'run_V3.1_b75f25h1015w0.5-0.50.5_balanced_10000.pkl'


    total_class_counter = {0: 0,
                           1: 0,
                           2: 0,
                           3: 0}
    sample_ids_per_class = {0: [],
                            1: [],
                            2: [],
                            3: []}

    num_sample_per_class = 4000
    total_dset = [[] for _ in range(8)]

    idx = 0
    for dir_path in data_dir_list:
        single_dset, single_indices = load_data(dir_path)

        size = len(single_dset[3])
        for sid in range(size):
            sample_ids_per_class[single_dset[3][sid]].append(idx)

            for attr_id in range(8):
                total_dset[attr_id].append(single_dset[attr_id][sid])

            idx += 1

        class_counter = collections.Counter(single_dset[3])
        print(class_counter)
        for k, v in class_counter.items():
            total_class_counter[k] += v

    print(total_class_counter)

    resampled_ids_16000 = {0: [],
                           1: [],
                           2: [],
                           3: []
                           }

    for k, v in sample_ids_per_class.items():
        rsp_list = random.sample(v, num_sample_per_class)
        resampled_ids_16000[k] = rsp_list
        print(k, len(rsp_list))

    num_sample_per_class_subset = 1500
    resampled_ids_6000 = []
    resampled_ids_10000 = []

    for k, v in resampled_ids_16000.items():
        rsp_list_subset = random.sample(v, num_sample_per_class_subset)
        resampled_ids_6000 += rsp_list_subset

        for id in v:
            if id not in rsp_list_subset:
                resampled_ids_10000.append(id)

    print(len(resampled_ids_6000), len(resampled_ids_10000))

    sampled_total_dset_6000 = [[] for _ in range(8)]
    sampled_total_dset_10000 = [[] for _ in range(8)]
    for attr_id in range(8):
        total_dset[attr_id] = np.array(total_dset[attr_id])

        sampled_total_dset_6000[attr_id] = total_dset[attr_id][resampled_ids_6000]
        sampled_total_dset_10000[attr_id] = total_dset[attr_id][resampled_ids_10000]


    with open(out_6000_path, 'wb') as f:
        pickle.dump(sampled_total_dset_6000, f)

    with open(out_full_path, 'wb') as f:
        pickle.dump(sampled_total_dset_10000, f)

    # fig, ax1 = plt.subplots()
    # b1, b2 = -6, 6
    # ax1.set_xlim(b1, b2)
    # ax1.set_ylim(b1, b2)
    #
    # color_map = {3: 'green', 0: 'blue', 1: 'brown', 2: 'red'}
    # for i in range(num_sample_per_class_subset*4):
    #     ax1.scatter(-sampled_total_dset_6000[0][i][0], -sampled_total_dset_6000[0][i][2],
    #                 color=color_map[sampled_total_dset_6000[3][i]], s=10)
    #
    # plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.savefig('w1.5.png', dpi=300)
    # # plt.show()


