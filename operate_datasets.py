import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from process_pickle import read_pickled_df
from net import digitize_signal, digitize_signal_1d


SGN_DGT_MODE_NAMES = ['sum', 'longitudal', 'projection', 'plane']
SGN_DGT_MODES = dict(zip(SGN_DGT_MODE_NAMES, 
                         np.arange(len(SGN_DGT_MODE_NAMES))))
# 0d - sum of all vals = 1 number, for baseline
# 1d - (z) = sum for each z, for advanced baseline
# 2d - (x,z)(y,z) variation = projections ala tomography, for release (no ghost hits)
# 3d - (x,y) variation, for conv testing


def make_dataset(pickled_TT_df, pickled_y_df, detector_params,
                 used_data_coef  = 1., 
                 sgn_dgt_mode    = SGN_DGT_MODES['sum']):

    assert len(pickled_TT_df) == len(pickled_y_df)
    assert used_data_coef > 0.
    assert used_data_coef <= 1.

    full_data_size = len(pickled_TT_df)

    # select subset of available data to make debug training faster
    data_size = int(used_data_coef * full_data_size)
    
    # normalised energies array cast to numpy and subset taken
    y_arr = pickled_y_df["E"].to_numpy()[:data_size]

    # create simplistic dataset (sum pixels in (x,y) representation)
    X_arr = []

    nb_of_plane = len(detector_params.snd_params[detector_params.configuration]["TT_POSITIONS"])
    
    for i in tqdm(range(data_size)):
        # xy variation
        
        shower_stat = None

        xy_plane = digitize_signal(pickled_TT_df.iloc[i], detector_params, filters = nb_of_plane)
        
        if   sgn_dgt_mode == SGN_DGT_MODES['plane']:
            # memory troubles!
            # this is not well written
            # be very carefull when using this
            shower_stat = xy_plane
        
        elif sgn_dgt_mode == SGN_DGT_MODES['projection']:
            xz = xy_plane.sum(axis=1)
            yz = xy_plane.sum(axis=2)
            
            shower_stat = np.concatenate((xz, yz), axis=1)

        elif sgn_dgt_mode == SGN_DGT_MODES['longitudal']:
            shower_stat = xy_plane.sum(axis=(1,2))

        elif sgn_dgt_mode == SGN_DGT_MODES['sum']:
            shower_stat = xy_plane.sum()
        
        else:
            raise Exception("Unknown signal digitization mode")

        X_arr.append(shower_stat)

    X_arr = np.array(X_arr)

    return X_arr, y_arr


def save_dataset(full_X, full_y, fname):
    with open(fname, 'wb') as file:
        np.savez(file, x=full_X, y=full_y)        
    return


def create_dataset(mode, detector_params, path_nuel, path_numu, path_nutau,
                   used_data_coef = 1.0):
    
    merged_TT_df, merged_y_full = read_pickled_df(detector_params, path_nuel, path_numu, path_nutau)

    assert mode in SGN_DGT_MODE_NAMES
    print(mode)
        
    dataset_fname = 'new_dataset_' + mode + '.npz'

    full_X, full_y = make_dataset(merged_TT_df, merged_y_full, detector_params,
                                  used_data_coef = used_data_coef,
                                  sgn_dgt_mode = SGN_DGT_MODES[mode])

    save_dataset(full_X, full_y, dataset_fname)
        
    return


def load_dataset(path, mode='sum'):
    assert mode in SGN_DGT_MODE_NAMES
    
    dataset_fname = os.path.join(os.path.expanduser(path), 
                                 'new_dataset_' + mode + '.npz')

    full_X, full_y = None, None

    # load if exists
    full_dts = None

    with open(dataset_fname, 'rb') as file:
        full_dts = np.load(file)#, allow_pickle=True)

        full_X = full_dts['x']
        full_y = full_dts['y']
        
    return full_X, full_y


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