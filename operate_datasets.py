import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from process_pickle import read_pickled_df

from net import digitize_signal_scifi
from net import digitize_signal_upstream_mu
from net import digitize_signal_downstream_mu


SGN_DGT_MODE_NAMES = ['sum', 'longitudal', 'projection', 'true'] # 'plane'
SGN_DGT_MODES = dict(zip(SGN_DGT_MODE_NAMES, 
                         np.arange(len(SGN_DGT_MODE_NAMES))))
# 0d - sum of all vals = 1 number, for baseline
# 1d - (z) = sum for each z, for advanced baseline
# 2d - (x,z)(y,z) variation = projections ala tomography, for release (no ghost hits)
# 3d - (x,y) variation, for conv testing


def detector_planes_num(det_params):
    nb_plane = dict()
    det_conf = det_params.snd_params[det_params.configuration]

    nb_plane['scifi']   = len(det_conf['SciFi_tracker']        ['TT_POSITIONS'])
    nb_plane['up_mu']   = len(det_conf['Mu_tracker_upstream']  ['TT_POSITIONS'])
    nb_plane['down_mu'] = len(det_conf['Mu_tracker_downstream']['TT_POSITIONS'])

    return nb_plane
    

def make_dataset(scifi_arr, mu_arr, en_arr, 
                 detector_params,
                 used_data_coef = 1., 
                 sgn_dgt_mode   = SGN_DGT_MODES['sum']):
    # check arrays validity
    assert len(scifi_arr) == len(mu_arr)
    assert len(scifi_arr) == len(en_arr)
    assert used_data_coef > 0.
    assert used_data_coef <= 1.

    full_data_size = len(scifi_arr)

    # select subset of available data to make debug training faster
    data_size = int(used_data_coef * full_data_size)
    
    # normalised energies array cast to numpy and subset taken
    y_arr = en_arr["E"].to_numpy()[:data_size]

    # create a simplistic dataset
    X_arr = []
    
    # get number of planes for each detector part
    filt_num = detector_planes_num(detector_params)
    
    for i in tqdm(range(data_size)):
        shower_stat = None
        
        scifi_resp   = digitize_signal_scifi        (scifi_arr.iloc[i],
                                                     detector_params, filt_num['scifi'])
        up_mu_resp   = digitize_signal_upstream_mu  (mu_arr.iloc[i], 
                                                     detector_params, filt_num['up_mu'])
        down_mu_resp = digitize_signal_downstream_mu(mu_arr.iloc[i], 
                                                     detector_params, filt_num['down_mu'])
        
        if sgn_dgt_mode == SGN_DGT_MODES['projection']:
            print(scifi_resp.shape, up_mu_resp.shape, down_mu_resp.shape)
            
            scifi_x  , scifi_y   = scifi_resp  .sum(axis=1), scifi_resp  .sum(axis=2)
            up_mu_x  , up_mu_y   = up_mu_resp  .sum(axis=1), scifi_resp  .sum(axis=2)
            doen_mu_x, down_mu_y = down_mu_resp.sum(axis=1), down_mu_resp.sum(axis=2)

            print(scifi_x.shape, scifi_y.shape)
            print(up_mu_x.shape, up_mu_y.shape)
            
            x_vec = np.concatenate((scifi_x, up_mu_x, doen_mu_x))
            y_vec = np.concatenate((scifi_y, up_mu_y, doen_mu_y))

            shower_stat = np.concatenate((x_vec, y_vec), axis=1)

        elif sgn_dgt_mode == SGN_DGT_MODES['longitudal']:
            num1 = scifi_resp.sum(axis=(1,2))
            num2 = up_mu_resp.sum(axis=(1,2))
            num3 = down_mu_resp.sum(axis=(1,2))

            shower_stat = np.concatenate((num1, num2, num3))
            
        elif sgn_dgt_mode == SGN_DGT_MODES['sum']:
            shower_stat = scifi_resp.sum() + up_mu_resp.sum() + down_mu_resp.sum()
            
        elif sgn_dgt_mode == SGN_DGT_MODES['true']:
            shower_stat = np.array([scifi_arr['X'][i].shape[0], 
                                    mu_arr   ['X'][i].shape[0]])
                      
        #elif sgn_dgt_mode == SGN_DGT_MODES['plane']:
        #    # memory troubles!
        #    # this is not well written
        #    # be very carefull when using this
        #    shower_stat = xy_plane
        
        else:
            raise Exception("Unknown signal digitization mode")

        X_arr.append(shower_stat)

    X_arr = np.array(X_arr)

    return X_arr, y_arr


def save_dataset(full_X, full_y, fname):
    with open(fname, 'wb') as file:
        np.savez(file, x=full_X, y=full_y)        
    return


def create_dataset(mode, detector_params, paths_dict, 
                   events_per_file, files_num, used_data_coef=1.0):
    scifi_arr, mu_arr, en_arr = read_pickled_df(detector_params, paths_dict, 
                                                events_per_file, files_num)

    assert mode in SGN_DGT_MODE_NAMES
    print(mode)
        
    dataset_fname = 'new_dataset_' + mode + '.npz'

    X_arr, y_arr = make_dataset(scifi_arr, mu_arr, en_arr, 
                                detector_params,
                                used_data_coef,
                                SGN_DGT_MODES[mode])

    save_dataset(X_arr, y_arr, dataset_fname)
        
    return


def load_dataset(path, mode='sum'):
    assert mode in SGN_DGT_MODE_NAMES
    
    dataset_fname = os.path.join(os.path.expanduser(path), 
                                 'new_dataset_' + mode + '.npz')

    X_arr, y_arr = None, None

    # load if exists
    full_dts = None

    with open(dataset_fname, 'rb') as file:
        packed_arr = np.load(file)#, allow_pickle=True)

        X_arr = packed_arr['x']
        y_arr = packed_arr['y']
        
    return X_arr, y_arr


def split_dataset(full_X, full_y, TRAIN_SIZE_RATIO = 0.9, RANDOM_SEED = 1543):
    data_size = full_X.shape[0]

    all_idx = np.arange(0, data_size)

    train_idx, test_idx, _, _ = train_test_split(all_idx, all_idx, 
                                                 train_size=TRAIN_SIZE_RATIO, 
                                                 random_state=RANDOM_SEED)

    train_size = len(train_idx)
    test_size  = len(test_idx)

    X_train = full_X[train_idx]
    y_train = full_y[train_idx]

    X_test = full_X[test_idx]
    y_test = full_y[test_idx]
    
    return X_train, y_train, X_test, y_test


def clip_dataset(X_arr, y_arr, min_clip):
    clip_idx = np.where(X_arr > min_clip)[0]

    X_arr_clip = X_arr[clip_idx]#.reshape(-1, 1)
    y_arr_clip = y_arr[clip_idx]#.reshape(-1, 1)

    return X_arr_clip, y_arr_clip