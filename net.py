import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_left
from utils import CM_TO_MUM
from coord_conv import CoordConv
import itertools

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, pool=False):
        super().__init__()
        self.block = nn.Sequential()
        self.block.add_module("conv", CoordConv(in_channels, out_channels,
                                                kernel_size=(k_size, k_size), stride=1, with_r=True))

        if pool:
            self.block.add_module("Pool", nn.MaxPool2d(2))
        self.block.add_module("BN", nn.BatchNorm2d(out_channels))
        self.block.add_module("Act", nn.ReLU())
        # self.block.add_module("dropout", nn.Dropout(p=0.5))

    def forward(self, x):
        return self.block(x)


class SNDNet(nn.Module):
    def __init__(self, n_input_filters):
        super().__init__()
        self.model = nn.Sequential(
            Block(n_input_filters, 32, pool=True),
            Block(32, 32, pool=True),
            Block(32, 64, pool=True),
            Block(64, 64, pool=True),
            Block(64, 64, pool=True),
            #Block(32, 32, pool=True),
            #Block(128, 128, pool=False),
            Flatten(),
            nn.Linear(1024, 1),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            #nn.Linear(1280,2)
        )

    def compute_loss(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        logits = self.model(X_batch)
        return F.mse_loss(logits, y_batch, reduction = 'none').mean()
        #return F.smooth_l1_loss(logits, y_batch).mean()

    def predict(self, X_batch):
        self.model.eval()
        return self.model(X_batch.to(self.device))

    @property
    def device(self):
        return next(self.model.parameters()).device


class MyDataset(Dataset):
    """
    Class defines how to preprocess data before feeding it into net.
    """
    def __init__(self, TT_df, y, parameters, data_frame_indices, n_filters):
        """
        :param TT_df: Pandas DataFrame of events
        :param y: Pandas DataFrame of true electron energy and distance
        :param parameters: Detector configuration
        :param data_frame_indices: Indices to train/test on
        :param n_filters: Number of TargetTrackers in the simulation
        """
        self.indices = data_frame_indices
        self.n_filters = n_filters
        self.X = TT_df
        self.y = y
        self.params = parameters

    def __getitem__(self, index):
        return torch.FloatTensor(digitize_signal(self.X.iloc[self.indices[index]],
                                                 self.params,
                                                 filters=self.n_filters)),\
               torch.FloatTensor(self.y.iloc[self.indices[index]])

    def __len__(self):
        return len(self.indices)


def digitize_signal_scifi(event, params, filters=1):
    """
    Convert pandas DataFrame to image
    :param event: Pandas DataFrame line
    :param params: Detector configuration
    :param filters: Number of TargetTrackers in the simulation
    :return: numpy tensor of shape (n_filters, H, W).
    """
    x_half = (params.snd_params[params.configuration]['SciFi_tracker']['X_max']-params.snd_params[params.configuration]['SciFi_tracker']['X_min'])/2
    y_half = (params.snd_params[params.configuration]['SciFi_tracker']['Y_max']-params.snd_params[params.configuration]['SciFi_tracker']['Y_min'])/2
    shape = (filters,
             int(np.ceil(y_half * 2 * CM_TO_MUM /
                         params.snd_params[params.configuration]['SciFi_tracker']["RESOLUTION"])),
             int(np.ceil(x_half * 2 * CM_TO_MUM /
                         params.snd_params[params.configuration]['SciFi_tracker']["RESOLUTION"])))
    response = np.zeros(shape)
    
    for x_index, y_index, z_pos in zip(np.floor((event['X'] + x_half) * CM_TO_MUM /
                                                params.snd_params[params.configuration]['SciFi_tracker']["RESOLUTION"]).astype(int),
                                       np.floor((event['Y'] + y_half) * CM_TO_MUM /
                                                params.snd_params[params.configuration]['SciFi_tracker']["RESOLUTION"]).astype(int),
                                       event['Z']):
        response[params.tt_map[bisect_left(params.scifi_tt_positions, z_pos)], y_index,x_index] += 1
        if response[params.tt_map[bisect_left(params.scifi_tt_positions, z_pos)], y_index, x_index] == 0:
            response[params.tt_map[bisect_left(params.scifi_tt_positions, z_pos)], y_index, x_index] += 1
    return response

def digitize_signal_downstream_mu(event, params, filters=1):
    """
    Convert pandas DataFrame to image
    :param event: Pandas DataFrame line
    :param params: Detector configuration
    :param filters: Number of TargetTrackers in the simulation
    :return: numpy tensor of shape (n_filters, H, W).
    """
    x_half = (params.snd_params[params.configuration]['Mu_tracker_downstream']['X_max']-params.snd_params[params.configuration]['Mu_tracker_downstream']['X_min'])/2
    y_half = (params.snd_params[params.configuration]['Mu_tracker_downstream']['Y_max']-params.snd_params[params.configuration]['Mu_tracker_downstream']['Y_min'])/2
    shape = (filters,
             int(np.ceil(y_half * 2 * CM_TO_MUM /
                         params.snd_params[params.configuration]['Mu_tracker_downstream']["RESOLUTION"])),
             int(np.ceil(x_half * 2 * CM_TO_MUM /
                         params.snd_params[params.configuration]['Mu_tracker_downstream']["RESOLUTION"])))
    response = np.zeros(shape)

    for x_index, y_index, z_pos in zip(np.floor((event['X'] + x_half) * CM_TO_MUM /
                                                params.snd_params[params.configuration]['Mu_tracker_downstream']["RESOLUTION"]).astype(int),
                                       np.floor((event['Y'] + y_half) * CM_TO_MUM /
                                                params.snd_params[params.configuration]['Mu_tracker_downstream']["RESOLUTION"]).astype(int),
                                       event['Z']):
        if (z_pos >  params.mu_downstream_tt_positions[0]) and (z_pos <  params.mu_downstream_tt_positions[-1]):
            response[params.tt_map[bisect_left(params.mu_downstream_tt_positions, z_pos)], y_index, x_index] = 1

    return response

def digitize_signal_upstream_mu(event, params, filters=1):
    """
    Convert pandas DataFrame to image
    :param event: Pandas DataFrame line
    :param params: Detector configuration
    :param filters: Number of TargetTrackers in the simulation
    :return: numpy tensor of shape (n_filters, H, W).
    """
    x_half = (params.snd_params[params.configuration]['Mu_tracker_upstream']['X_max']-params.snd_params[params.configuration]['Mu_tracker_upstream']['X_min'])/2
    y_half = (params.snd_params[params.configuration]['Mu_tracker_upstream']['Y_max']-params.snd_params[params.configuration]['Mu_tracker_upstream']['Y_min'])/2
    shape = (filters,
             int(np.ceil(y_half * 2 * CM_TO_MUM / params.snd_params[params.configuration]['Mu_tracker_upstream']["RESOLUTION"])),
             1)
    response = np.zeros(shape)

    for y_index, z_pos in zip(np.floor((event['Y'] + y_half) * CM_TO_MUM / params.snd_params[params.configuration]['Mu_tracker_upstream']["RESOLUTION"]).astype(int),
                                       event['Z']):
        if (z_pos >  params.mu_upstream_tt_positions[0]) and (z_pos <  params.mu_upstream_tt_positions[-1]):
            response[params.tt_map[bisect_left(params.mu_upstream_tt_positions, z_pos)], y_index, 0] = 1

    return response
