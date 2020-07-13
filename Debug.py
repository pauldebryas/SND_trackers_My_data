#!/usr/bin/env python3
# Import Class from utils.py & net.py file
from utils import DataPreprocess, Parameters
from net import SNDNet, MyDataset, digitize_signal
# usful module 
import torch
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
from IPython import display
import os
import gc # Gabage collector interface (to debug stuff)

# Test to see if cuda is available or not + listed the CUDA devices that are available
try:
    assert(torch.cuda.is_available())
except:
    raise Exception("CUDA is not available")
n_devices = torch.cuda.device_count()
print("\nWelcome!\n\nCUDA devices available:\n")
for i in range(n_devices):
    print("\t{}\twith CUDA capability {}".format(torch.cuda.get_device_name(device=i), torch.cuda.get_device_capability(device=i)))
print("\n")
device = torch.device("cuda", 0)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Turn off interactive plotting: for long run it makes screwing up everything
plt.ioff()

# Here we choose the geometry with 9 time the radiation length
params = Parameters("4X0")  #!!!!!!!!!!!!!!!!!!!!!CHANGE THE DIMENTION !!!!!!!!!!!!!!!!
processed_file_path = os.path.expandvars("$HOME/DS4red/ship_tt_processed_data") #!!!!!!!!!!!!!!!!!!!!!CHANGE THE PATH !!!!!!!!!!!!!!!!
step_size = 5000    # size of a chunk
file_size = 240000  # size of the BigFile.root file
n_steps = int(file_size / step_size) # number of chunks

# ------------------------------------------ LOAD THE reindex_TT_df & reindex_y_full PD.DATAFRAME --------------------------------------------------------------------------

chunklist_TT_df = []  # list of the TT_df file of each chunk
chunklist_y_full = [] # list of the y_full file of each chunk

# It is reading and analysing data by chunk instead of all at the time (memory leak problem)
print("\nReading the tt_cleared_reduced.pkl & y_cleared.pkl files by chunk")
#First 2
outpath = processed_file_path + "/{}".format(0)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
outpath = processed_file_path + "/{}".format(1)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

reindex_TT_df = pd.concat([chunklist_TT_df[0],chunklist_TT_df[1]],ignore_index=True)
reindex_y_full = pd.concat([chunklist_y_full[0],chunklist_y_full[1]], ignore_index=True)

for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
    outpath = processed_file_path + "/{}".format(i+2)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl"))) # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
    reindex_TT_df = pd.concat([reindex_TT_df,chunklist_TT_df[i+2]], ignore_index=True)
    reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

# reset to empty space
chunklist_TT_df = []
chunklist_y_full = []
nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])


# True value of NRJ/dist for each true electron event
#y = reindex_y_full[["E", "Z","THETA"]]
y = reindex_y_full[["E"]]
NORM = 1. / 100
y["E"] *= NORM
#y["Z"] *= -1
#y["THETA"] *= (180/np.pi)


# reset to empty space
#reindex_y_full = []

# Spliting
print("\nSplitting the data into a training and a testing sample")

indeces = np.arange(len(reindex_TT_df))
train_indeces, test_indeces, _, _ = train_test_split(indeces, indeces, train_size=0.9, random_state=1543)

test_indeces = np.arange(10000)

#batch_size = 512
batch_size = 150

test_dataset = MyDataset(reindex_TT_df, y, params, test_indeces, n_filters=nb_of_plane)
test_batch_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# reset to empty space
reindex_TT_df=[]

# Saving the true Energy for the test sample
True_value = y["E"][test_indeces]
True_test_value = True_value.to_numpy()

net = torch.load("../DS4red/9X0_file/49_9X0_coordconv.pt")
preds = []
with torch.no_grad():
    for (X_batch, y_batch) in test_batch_gen:
        preds.append(net.predict(X_batch))

ans = np.concatenate([p.detach().cpu().numpy() for p in preds])
Pred_test_value = ans[:, 0]


#------------------------------------Parameters-------------------------------------
#Training epoch of the network
epoch = 49

#filename where to find datas and save the figures
simulation_name = "DS4"

#maximum energy of the shooted electron
NRJ_max = 200

#number of energy bin
number_of_bin = 5

#number of bin in the histograms to compute mu/sigma for each NRJ range
nbins = 75

#renormalise (make sur it is coherent with run_script value)
NORM = 1. / 100
True_test_value = True_test_value / NORM
Pred_test_value = Pred_test_value / NORM

bin_size= NRJ_max/number_of_bin


#create index/deltaE = [[],[],[],[],[],[],[],[]] (size of the number of bin)
index = []
deltaE = []
for i in range(0,number_of_bin):
    index.append([])
    deltaE.append([])

for i in range(len(True_test_value)):
    for j in range(number_of_bin):
        if True_test_value[i] > (bin_size*j) and True_test_value[i] < (bin_size*(j+1)):
            index[j].append(i)
            break

for i in range(number_of_bin):
    deltaE[i] = np.take(True_test_value,index[i])-np.take(Pred_test_value,index[i])


