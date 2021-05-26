#Preprocessing_script.py

# This file is use to transform the raw data file (root file) to processed data (pickle file), in which only the needed informations are stored.

# -----------------------------------IMPORT AND OPTIONS FOR PREPROCESSING-------------------------------------------------------------
# Import Class from utils.py & net.py file
from utils import DataPreprocess, Parameters
from net import digitize_signal_scifi
from net import digitize_signal_mu

# Import usfull module 
from matplotlib import pylab as plt
import numpy as np
import root_numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
from IPython import display
import os
import argparse
from ROOT import TH1F, TFile

# -----------------------------------Reading the pickle files by chunk ---------------------------------------------------------------
def reading_pkl_files(n_steps,processed_file_path):
    chunklist_y_full = [] # list of the y_full file of each chunk

    # It is reading and analysing data by chunk instead of all at the same time (memory leak problem)
    #First 2 
    outpath = processed_file_path + "/{}".format(0)
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
    outpath = processed_file_path + "/{}".format(1)
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

    reindex_y_full = pd.concat([chunklist_y_full[0],chunklist_y_full[1]], ignore_index=True)

    for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
        outpath = processed_file_path + "/{}".format(i+2)
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
        reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

    #reset empty space
    chunklist_y_full = []

    return reindex_y_full

# ----------------------------------------------------Paramaters----------------------------------------------------------------------
# parameter of the geometry configuration
params = Parameters("SNDatLHC") 

# Path to the raw Data root file and the localisation of where to store the pickle file

loc_of_pkl_file = "/home/debryas/data/processed_data/nhits0_5/"
loc_of_pkl_file1 = "/home/debryas/data/processed_data/CCDIS/"
processed_file_path = os.path.expandvars(loc_of_pkl_file)
processed_file_path1 = os.path.expandvars(loc_of_pkl_file1)

step_size = 1000
file_size = 5000
n_steps = int(file_size / step_size)

reindex_y_full = reading_pkl_files(n_steps,processed_file_path)
reindex_y_full1 = reading_pkl_files(n_steps,processed_file_path1)

less  = reindex_y_full['E'].to_numpy()
above = reindex_y_full1['E'].to_numpy()

print('Number of events for low hits:  '+ str(len(less)))
print('Number of events for above 5 hits:  '+ str(len(above)))

#maximum energy of the shooted particle
NRJ_max = 5000

NRJ_min = 0

#number of energy bin
number_of_bin = 4

#bins
#bins_adap = [[0,400],[400,700],[700,1200],[1200,5000]]
#bins_adap = [[0,1000],[1000,2000],[2000,3000],[3000,5000]]
bins_adap = [[0,500],[500,1000],[1000,2000],[2000,5000]]

bins_width = []
bins_mean = []
less_bin = []
above_bin = []
for i in range(0,number_of_bin):
    less_bin.append([])
    above_bin.append([])
    bins_mean.append(bins_adap[i][0]+(bins_adap[i][1]-bins_adap[i][0])/2)
    bins_width.append(bins_adap[i][1]-bins_adap[i][0]-10)
 
for i in range(len(less)):
    for j in range(number_of_bin):
        if less[i] > bins_adap[j][0] and less[i] < bins_adap[j][1]:
            less_bin[j].append(less[i])
            break

for i in range(len(above)):
    for j in range(number_of_bin):
        if above[i] > bins_adap[j][0] and above[i] < bins_adap[j][1]:
            above_bin[j].append(above[i])
            break

ratio = []
for i in range(number_of_bin):
    ratio.append(len(less_bin[i])/len(above_bin[i]))

print(bins_mean)
print(bins_width)
print(ratio)
ax = plt.bar(bins_mean, ratio, bins_width)
#ax.set(xlabel ='E bin [GeV]', ylabel ='ratio',title ='low/high number of hits')
plt.show()
