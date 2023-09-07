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

# import xgboost as xgb
import lightgbm as lgb
import numpy as np
import torch
import yaml
from geomloss import SamplesLoss
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split


from nets.trainer import Trainer
from sim_objects import Land

from sklearn.neural_network import MLPRegressor
# warnings.simplefilter("ignore")


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def hellinger(p, q):
#     """Hellinger distance between two discrete distributions.
#        In pure Python.
#        Source: https://nbviewer.org/gist/Teagum/460a508cda99f9874e4ff828e1896862
#     """
#     return sum([(math.sqrt(t[0])-math.sqrt(t[1]))*(math.sqrt(t[0])-math.sqrt(t[1])) \
#                 for t in zip(p,q)])/math.sqrt(2.)


def drift_analysis(d1, d2, loss):
    """
    Source: https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html#sphx-glr-auto-examples-sinkhorn-multiscale-plot-transport-blur-py
    """
    Loss =  SamplesLoss(loss, p=2, blur=0.05) #sinkhorn is Wasserstein Distance. blur is a paramter
    joint_dist = torch.tensor(d1)
    joint_dist2 = torch.tensor(d2)

    return Loss(joint_dist, joint_dist2).item()


class Processor:
    """
    It achieves several functions:
    1) load the raw data files generated from the simulation.
    2) train, validate, and test the self-labeling method under the nested cross-validation loop.
    3) generate npy files of training(lb, slb), val, and test set for testing on TorchSSL.

    """
    def __init__(self, args):

        '''
        output dir structure
        --out_dir
            --out_data_dir_name
                --result.txt file
                --npy folder
        '''
        data_dir_name = os.path.basename(args.data_path)
        if args.add_data:
            data_dir_name += '_addi_25'
        print('data dirt name: ', data_dir_name)
        args.data_dir_name = data_dir_name
        args.out_dir_path = os.path.join(args.out_dir_path, args.data_dir_name)
        if not os.path.exists(args.out_dir_path):
            os.makedirs(args.out_dir_path)

        result_file_name = 'results_x{}z{}t{}_{}'.format(str(args.x_offset_vel),
                                                         str(args.z_offset_vel),
                                                         '1',
                                                         args.data_dir_name)
        print('use pretrain: ', args.use_pretrain)
        if not args.use_pretrain:
            result_file_name += '__nopretrain'
        result_file_name += '.txt'
        args.result_file_path = os.path.join(args.out_dir_path, result_file_name)

        args.work_dir = os.path.join(args.out_dir_path, result_file_name)
        # load land config and create land objects.
        # below is hardcoded center positions of each land block.
        self.land_centers = [[0, 5.0, -5.0], [1, 5.0, -3.0], [2, 5.0, -1.0], [3, 5.0, 1.0],
                        [4, 5.0, 3.0], [5, 5.0, 5.0], [6, 3.0, -5.0], [7, 3.0, -3.0],
                        [8, 3.0, -1.0], [9, 3.0, 1.0], [10, 3.0, 3.0], [11, 3.0, 5.0],
                        [12, 1.0, -5.0], [13, 1.0, -3.0], [14, 1.0, -1.0], [15, 1.0, 1.0],
                        [16, 1.0, 3.0], [17, 1.0, 5.0], [18, -1.0, -5.0], [19, -1.0, -3.0],
                        [20, -1.0, -1.0], [21, -1.0, 1.0], [22, -1.0, 3.0], [23, -1.0, 5.0],
                        [24, -3.0, -5.0], [25, -3.0, -3.0], [26, -3.0, -1.0], [27, -3.0, 1.0],
                        [28, -3.0, 3.0], [29, -3.0, 5.0], [30, -5.0, -5.0], [31, -5.0, -3.0],
                        [32, -5.0, -1.0], [33, -5.0, 1.0], [34, -5.0, 3.0], [35, -5.0, 5.0]]

        # for file in os.listdir(args.data_path):
            # if file.endswith(".yaml"):
            #     land_params_yaml_path = os.path.join(args.data_path, file)
        land_params_yaml_path = args.land_params_yaml_path

        with open(land_params_yaml_path) as f:
            land_args = yaml.full_load(f)
        self.lands = []
        for i in range(len(land_args['params'])):
            land = Land(**land_args['params'][i])
            self.lands.append(land)

        self.args = args
        print('result file path: ', self.args.result_file_path)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.args.print_log:
            with open(self.args.result_file_path, 'a') as f:
                print(s, file=f)

    def load_data(self, path, isPlot=False):
        with open(path, 'rb') as f:
            dset = pickle.load(f)
        indices = list(range(len(dset[0])))

        return dset, np.array(indices)

    def cross_validation(self, indices, pretrain_dataset, dataset, add_dset=None, add_indices=None):

        x_vel = self.args.x_offset_vel
        z_vel = self.args.z_offset_vel

        kf = StratifiedKFold(n_splits=3, random_state=101, shuffle=True)
        kf_lb_slb = StratifiedKFold(n_splits=5, random_state=102, shuffle=True)


        pretrain_cls_score = []
        itm_score = []
        itm_mae = []
        slb_score = collections.defaultdict(list)
        gt_score = collections.defaultdict(list)
        semi_score = collections.defaultdict(list)

        x_offset_list = []
        z_offset_list = []
        dist = []

        npy_path = os.path.join(self.args.out_dir_path, '{}_npy'.format(self.args.data_dir_name))

        outer_fold_idx = 0
        # for i in range(len(indices)):
        #     print(indices[i], dataset[3][indices][i])

        for train_index, test_index in kf.split(indices, dataset[3][indices]):

            print('*'*30 + 'Outer fold {} started'.format(str(outer_fold_idx)) + '*'*30)

            testset_indices = indices[test_index]

            input_pos_test = dataset[0][testset_indices]
            output_pos_test = dataset[1][testset_indices]
            duration_test = dataset[2][testset_indices]
            classes_test = dataset[3][testset_indices]
            print("test set size: ", len(input_pos_test))

            ########################################
            fold_path = os.path.join(npy_path, 'fold_{}'.format(outer_fold_idx))
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            np.save(os.path.join(fold_path, 'data_test_{}.npy'.format(outer_fold_idx)), input_pos_test)
            with open(os.path.join(fold_path, 'label_test_{}.pkl'.format(outer_fold_idx)), 'wb') as f:
                pickle.dump(classes_test, f)
            ########################################

            # 1000
            trainset_indices = indices[train_index]

            inner_fold_idx = 0
            for slb_index, pre_index in kf_lb_slb.split(trainset_indices, dataset[3][trainset_indices]):
                print('*'*30 + 'Inner fold {} started'.format(str(inner_fold_idx)) + '*'*30)

                pre_indices = trainset_indices[pre_index]
                input_pos_pre = pretrain_dataset[0][pre_indices]
                output_pos_pre = pretrain_dataset[1][pre_indices]
                duration_pre = pretrain_dataset[2][pre_indices]
                classes_pre = pretrain_dataset[3][pre_indices]
                pos_arrs_pre = pretrain_dataset[4][pre_indices]
                time_idx_arrs_pre = pretrain_dataset[5][pre_indices]

                # self-label datsaet
                slb_indices = trainset_indices[slb_index]
                input_pos_slb = dataset[0][slb_indices]
                output_pos_slb = dataset[1][slb_indices]
                duration_slb = dataset[2][slb_indices]
                classes_slb = dataset[3][slb_indices]
                pos_arrs_slb = dataset[4][slb_indices]
                time_idx_arrs_slb = dataset[5][slb_indices]
                raw_output_pos_slb = dataset[7][slb_indices]

                subset_size = 500
                total_num_ssets = 5

                # process additional dataset
                if self.args.add_data:
                    reordered_add_indices = []
                    to_split_indices = add_indices
                    for _ in range(int(len(to_split_indices) / subset_size) - 1):
                        curr_size = len(to_split_indices)
                        ratio = subset_size / curr_size
                        to_split_indices, selected_add_indices = train_test_split(to_split_indices,
                                                                                  test_size=ratio,
                                                                                  shuffle=True,
                                                                                  random_state=0,
                                                                                  stratify=add_dset[3][to_split_indices])
                        reordered_add_indices += selected_add_indices.tolist()
                    reordered_add_indices += to_split_indices.tolist()

                    input_pos_slb = np.concatenate((input_pos_slb[:subset_size*5],
                                                    add_dset[0][reordered_add_indices],
                                                    input_pos_slb[subset_size * 5:]
                                                    ))
                    output_pos_slb = np.concatenate((output_pos_slb[:subset_size*5],
                                                     add_dset[1][reordered_add_indices],
                                                     output_pos_slb[subset_size * 5:]
                                                     ))
                    duration_slb = np.concatenate((duration_slb[:subset_size*5],
                                                   add_dset[2][reordered_add_indices],
                                                   duration_slb[subset_size * 5:]
                                                   ))
                    classes_slb = np.concatenate((classes_slb[:subset_size*5],
                                                  add_dset[3][reordered_add_indices],
                                                  classes_slb[subset_size * 5:]
                                                  ))
                    pos_arrs_slb = np.concatenate((pos_arrs_slb[:subset_size*5],
                                                   add_dset[4][reordered_add_indices],
                                                   pos_arrs_slb[subset_size * 5:]
                                                   ))
                    time_idx_arrs_slb = np.concatenate((time_idx_arrs_slb[:subset_size*5],
                                                        add_dset[5][reordered_add_indices],
                                                        time_idx_arrs_slb[subset_size * 5:]
                                                        ))
                    raw_output_pos_slb = np.concatenate((raw_output_pos_slb[:subset_size*5],
                                                         add_dset[7][reordered_add_indices],
                                                         raw_output_pos_slb[subset_size * 5:]
                                                         ))

                print('shapes: ', input_pos_slb.shape, output_pos_slb.shape, classes_slb.shape,
                      pos_arrs_slb.shape, time_idx_arrs_slb.shape, raw_output_pos_slb.shape)

                if self.args.add_data:
                    input_pos_val = input_pos_slb[subset_size*total_num_ssets:]
                    classes_val = classes_slb[subset_size*total_num_ssets:]
                else:
                    input_pos_val = input_pos_slb[subset_size * 5:]
                    classes_val = classes_slb[subset_size * 5:]

                inside_fold_path = os.path.join(fold_path, 'training_fold_{}'.format(inner_fold_idx))
                if not os.path.exists(inside_fold_path):
                    os.makedirs(inside_fold_path)
                np.save(os.path.join(inside_fold_path, 'data_lb_{}.npy'.format(inner_fold_idx)), input_pos_pre)
                with open(os.path.join(inside_fold_path, 'label_lb_{}.pkl'.format(inner_fold_idx)), 'wb') as f:
                    pickle.dump(classes_pre, f)

                np.save(os.path.join(inside_fold_path, 'data_val_{}.npy'.format(inner_fold_idx)), input_pos_val)
                with open(os.path.join(inside_fold_path, 'label_val_{}.pkl'.format(inner_fold_idx)), 'wb') as f:
                    pickle.dump(classes_val, f)

                # interaction time model (ITM) pretraining and testing
                itm = lgb.LGBMRegressor(boosting_type='dart', n_estimators=500, random_state=12,
                                        drop_rate=0.1, reg_lambda=0)

                print(duration_pre[0], output_pos_pre[0])
                itm.fit(output_pos_pre, duration_pre)
                itm_train_score = itm.score(output_pos_pre, duration_pre)
                itm_train_mae = mean_absolute_error(duration_pre, itm.predict(output_pos_pre))
                itm_test_score = itm.score(output_pos_test, duration_test)
                itm_test_mae = mean_absolute_error(duration_test, itm.predict(output_pos_test))
                print('gbdt train score: ', itm_train_score, 'mae: ', itm_train_mae)
                print('gbdt test score: ', itm_test_score, itm_test_mae)
                # print('feature importance: ', itm.feature_importances_)
                itm_mae.append(itm_test_mae)
                itm_score.append(itm_test_score)

                for slb_i in range(1, total_num_ssets+1):
                    lo, hi = 0, slb_i * subset_size

                    np.save(os.path.join(inside_fold_path,
                            'data_slb_{a}_{b}.npy'.format(a=inner_fold_idx, b=slb_i*subset_size)), input_pos_slb[:hi])
                    with open(os.path.join(inside_fold_path, 'label_slb_{a}_{b}.pkl'.format(a=inner_fold_idx, b=slb_i*subset_size)), 'wb') as f:
                        pickle.dump(classes_slb[:hi], f)

                    # an independent pretrain of the slb classifier to get the pretrain score
                    # if self.args.use_pretrain:
                    #     init_seed(self.args.seed)
                    #     cls_pre = Trainer(self.args)
                    #     cls_pre.load_data(input_pos_pre,
                    #                    classes_pre,
                    #                    input_pos_test,
                    #                    classes_test,
                    #                       input_pos_val,
                    #                       classes_val)
                    #     cls_pre.start()
                    #     cls_pre_score = cls_pre.best_acc[0]
                    # else:
                    #     cls_pre_score = 0
                    #
                    # pretrain_cls_score.append(cls_pre_score)
                    # print('initial cls score: ', cls_pre_score)


                    # Conduct self-labeling based on trained ITM and generate self-labeled
                    # data-label pairs.
                    mislabeled = 0
                    gt_input_pos, self_labeled_input_pos, self_labeled_labl = [], [], []
                    for di in range(hi):
                        input_effect_regr = output_pos_slb[di]  # get the ITM input data
                        pred_time = itm.predict(input_effect_regr.reshape(1, -1))  # ITM inference
                        pred_time = int(sum(pred_time)/len(pred_time))  # take the integer part of inferred interaction time.

                        x_offset = 0    # offset along x direction
                        z_offset = 0    # offset along z direction
                        maxtime = time_idx_arrs_slb[di][-1] # get the maximum timestamp of the simulation
                        pred_start_time = maxtime - pred_time   # backtrack to the inferred start time where the input data to be selected.

                        # add the horizontal offset if needed
                        if pred_start_time < time_idx_arrs_slb[di][0]:
                            time_penalty = time_idx_arrs_slb[di][0] - pred_start_time
                            pred_start_time = time_idx_arrs_slb[di][0]

                            # time_penalty = round(time_penalty / 10) * 10 // 1  # another way to penalize inaccurate ITM.
                            x_sign = 1 if rng.random() >= 0.5 else -1   # get the random direction
                            z_sign = 1 if rng.random() >= 0.5 else -1

                            # x_vel_actual = x_vel
                            # z_vel_actual = z_vel
                            x_vel_actual = rng.normal(x_vel, x_vel / 3)
                            z_vel_actual = rng.normal(z_vel, z_vel / 3)

                            x_offset = x_sign * (time_penalty*x_vel_actual)
                            z_offset = z_sign * (time_penalty*z_vel_actual)

                            x_offset_list.append(abs(x_offset))
                            z_offset_list.append(abs(z_offset))
                            dist.append(np.sqrt(x_offset ** 2 + z_offset ** 2))

                        # alternatively, index_for_pos can be added with 1 to avoid some simulation
                        # errors in the first position.
                        # index_for_pos = time_idx_arrs_slb[di].index(pred_start_time) + 1
                        index_for_pos = time_idx_arrs_slb[di].index(pred_start_time)

                        selected_pos = pos_arrs_slb[di][index_for_pos]
                        selected_pos[0] += x_offset
                        selected_pos[2] += z_offset

                        final_pos_x, final_pos_z = raw_output_pos_slb[di][0], raw_output_pos_slb[di][2]
                        if final_pos_x > 6 or final_pos_x < -6 or final_pos_z > 6 or final_pos_z < -6:
                            label = 3
                        else:
                            dist_sort = sorted(self.land_centers, key=lambda x: (x[1]-final_pos_x)**2+(x[2]-final_pos_z)**2) # (id, x, z)
                            closest_land_idx = dist_sort[0][0]
                            label = self.lands[closest_land_idx].category
                        if label != classes_slb[di]:
                            mislabeled += 1

                        self_labeled_input_pos.append(selected_pos)
                        gt_input_pos.append(input_pos_slb[di])
                        self_labeled_labl.append(label)


                    # self-labeling training and test. Combined with pre set or no.
                    init_seed(self.args.seed)
                    cls_slb = Trainer(self.args)
                    if self.args.use_pretrain:
                        boost_train_set = np.concatenate((input_pos_pre, np.array(self_labeled_input_pos)))
                        boost_train_label = np.concatenate((classes_pre, np.array(self_labeled_labl)))
                    else:
                        boost_train_set = np.array(self_labeled_input_pos)
                        boost_train_label = np.array(self_labeled_labl)

                    cls_slb.load_data(boost_train_set, boost_train_label,
                                      input_pos_test, classes_test,
                                    input_pos_val,
                                    classes_val)
                    cls_slb.start()
                    slb_test_score = cls_slb.best_acc[0]
                    slb_score[slb_i].append(slb_test_score)


                    # fully supervised training and test.
                    init_seed(self.args.seed)     # re-init the random see
                    cls_fs = Trainer(self.args)     # init a new fs classifier
                    if self.args.use_pretrain:
                        cls_fs.load_data(np.concatenate((input_pos_pre, input_pos_slb[:hi])),
                                       np.concatenate((classes_pre, classes_slb[:hi])),
                                       input_pos_test,
                                       classes_test,
                                       input_pos_val,
                                       classes_val)
                    else:
                        cls_fs.load_data(input_pos_slb[:hi],
                                       classes_slb[:hi],
                                       input_pos_test,
                                       classes_test,
                                       input_pos_val,
                                       classes_val)
                    cls_fs.start()
                    gt_score[slb_i].append(cls_fs.best_acc[0])
                    del cls_fs

                    print('$'*60)

                inner_fold_idx += 1
                print('*'*30 + 'Inner fold {} finished'.format(str(inner_fold_idx-1)) + '*'*30)

            outer_fold_idx += 1
            print('*'*30 + 'Outer fold {} finished'.format(str(outer_fold_idx-1)) + '*'*30)


        self.print_log(f'Data path: {self.args.data_path}')
        if pretrain_cls_score:
            self.print_log(f'pretrain_cls_score: {pretrain_cls_score}')
            self.print_log(f'averaged pretrain cls score: {sum(pretrain_cls_score)/len(pretrain_cls_score)}')
        self.print_log(f'ITM score list: {itm_score}')
        if itm_score:
            self.print_log(f'averaged ITM score: {sum(itm_score)/len(itm_score)}')
        self.print_log(f'ITM MAE list: {itm_mae}')
        if itm_mae:
            self.print_log(f'averaged ITM MAE: {sum(itm_mae)/len(itm_mae)}')
        self.print_log(f'slb score list: {slb_score}')
        self.print_log(f'gt score list: {gt_score}')

        self.print_log(f'slb ave scores:')
        if slb_score:
            for slb_i in range(1, max(slb_score) + 1):
                if slb_score and slb_score[slb_i]:
                    self.print_log(f'{slb_i*100} data: {sum(slb_score[slb_i]) / len(slb_score[slb_i])}')
        self.print_log(f'gt ave scores:')
        if gt_score:
            for slb_i in range(1, max(gt_score) + 1):
                if gt_score and gt_score[slb_i]:
                    self.print_log(f'{slb_i*100} data: {sum(gt_score[slb_i]) / len(gt_score[slb_i])}')

        if x_offset_list:
            self.print_log(f'averaged x offset: {sum(x_offset_list) / len(x_offset_list)}')
        if z_offset_list:
            self.print_log(f'averaged z offset: {sum(z_offset_list) / len(z_offset_list)}')
        if dist:
            self.print_log(f'averaged offset distance: {sum(dist) / len(dist)}')


    def measure_distance(self, d1, d2):
        input_pos = d1[0]
        input_pos_pert = d2[0]
        classes = d1[3].reshape((len(d1[3]), 1))
        classes_pert = d2[3].reshape((len(d2[3]), 1))

        p = np.concatenate((input_pos, classes), axis=1)
        q = np.concatenate((input_pos_pert, classes_pert), axis=1)

        self.print_log(f"Gaussian MMD distance w/ wind: {drift_analysis(p, q, 'gaussian')}")
        self.print_log(f"Energy distance w/ wind: {drift_analysis(p, q, 'energy')}")
        self.print_log(f"Sinkhorn distance w/ wind: {drift_analysis(p, q, 'sinkhorn')}")


    def process(self):
        # load datasets
        program_start_time = datetime.now()
        self.print_log(f"Start Time: {program_start_time}")
        pretrain_dataset, pre_indices = self.load_data(self.args.pretrain_path)
        dataset, indices = self.load_data(self.args.data_path)
        if self.args.add_data:
            add_dset, add_indices = self.load_data(self.args.add_data)


        # init seed and shuffle the data indices
        init_seed(self.args.seed)
        rng.shuffle(indices)
        if self.args.add_data:
            rng_add.shuffle(add_indices)

        # run the training and test in cross validation.
        if not self.args.add_data:
            self.cross_validation(indices, pretrain_dataset, dataset)
        else:
            self.cross_validation(indices, pretrain_dataset, dataset, add_dset, add_indices)

        # measure the distance with the disturbed dataset
        self.measure_distance(pretrain_dataset, dataset)

        program_end_time = datetime.now()
        self.print_log(f"End Time: {program_end_time}")
        self.print_log(f"Execution Time: {program_end_time - program_start_time}")


if __name__ == '__main__':

    rng = np.random.RandomState(7)
    rng_add = np.random.RandomState(7)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', type=str, default='', required=True)
    parser.add_argument('--data_path', type=str, default='', required=True)
    parser.add_argument('--out_dir_path', type=str, default='', required=True)
    parser.add_argument('--land_params_yaml_path', type=str, required=False)
    parser.add_argument('--add_data', type=str, default='', required=False)


    parser.add_argument('--use_pretrain', type=int, default=1, required=False)
    parser.add_argument('--x_offset_vel', type=float, default=0.05, required=False)
    parser.add_argument('--z_offset_vel', type=float, default=0.05, required=False)


    # load MLP network args
    with open('nets/net_config.yaml', 'r') as f:
        net_config = yaml.full_load(f)
    parser.set_defaults(**net_config)
    args = parser.parse_args()

    p = Processor(args)
    p.process()


