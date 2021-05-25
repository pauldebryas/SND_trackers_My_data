#Preprocessing_script.py

# This file is use to transform the raw data file (root file) to processed data (pickle file), in which only the needed informations are stored.

# -----------------------------------IMPORT AND OPTIONS FOR PREPROCESSING-------------------------------------------------------------
# Import Class from utils.py & net.py file
from utils import DataPreprocess, Parameters
from net import digitize_signal_scifi
from net import digitize_signal_upstream_mu
from net import digitize_signal_downstream_mu

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
    chunklist_TT_df = []  # list of the TT_df file of each chunk
    chunklist_Mu_df = []  # list of the TT_df file of each chunk
    chunklist_y_full = [] # list of the y_full file of each chunk

    # It is reading and analysing data by chunk instead of all at the same time (memory leak problem)
    #First 2 
    outpath = processed_file_path + "/{}".format(0)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
    chunklist_Mu_df.append(pd.read_pickle(os.path.join(outpath, "mu_cleared.pkl")))
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
    outpath = processed_file_path + "/{}".format(1)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
    chunklist_Mu_df.append(pd.read_pickle(os.path.join(outpath, "mu_cleared.pkl")))
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

    reindex_TT_df = pd.concat([chunklist_TT_df[0],chunklist_TT_df[1]],ignore_index=True)
    reindex_Mu_df = pd.concat([chunklist_Mu_df[0],chunklist_Mu_df[1]],ignore_index=True)
    reindex_y_full = pd.concat([chunklist_y_full[0],chunklist_y_full[1]], ignore_index=True)

    for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
        outpath = processed_file_path + "/{}".format(i+2)
        chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl"))) # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        chunklist_Mu_df.append(pd.read_pickle(os.path.join(outpath, "mu_cleared.pkl"))) # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
        reindex_TT_df = pd.concat([reindex_TT_df,chunklist_TT_df[i+2]], ignore_index=True)
        reindex_Mu_df = pd.concat([reindex_Mu_df,chunklist_Mu_df[i+2]], ignore_index=True)
        reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

    #reset empty space
    chunklist_TT_df = []
    chunklist_Mu_df = []
    chunklist_y_full = []

    return reindex_TT_df, reindex_Mu_df, reindex_y_full

# ----------------------------------------------------Paramaters----------------------------------------------------------------------
# parameter of the geometry configuration
params = Parameters("SNDatLHC") 

# Path to the raw Data root file and the localisation of where to store the pickle file
filename = "$HOME/data/heavy_data/numu_400kevents.root"
loc_of_pkl_file = "$HOME/data/processed_data/numu/"
processed_file_path = os.path.expandvars(loc_of_pkl_file)

# Usualy, Data file is too large to be read in one time, that the reason why we read it by "chunk" of step_size events
# number of event in a chunk
step_size = 4000
# number of events to be analyse. Warning: Maximum number should not exceed the maximum number of events in the root file it must be a multiple of step_size
file_size = 380000
#file number to begging with. Useful when the raw data file is split in multiple file. Default value must be 0.
begining_file = 0

#index of the event you want to plot the signal
index=10 # Warning: this should not exceed the maximum number of events proceed. 

# ----------------------------------------------------Global variables----------------------------------------------------------------
n_steps = int(file_size / step_size) 
nb_of_scifi_plane = len(params.snd_params[params.configuration]['SciFi_tracker']['TT_POSITIONS'])
nb_of_upstream_mu_plane = len(params.snd_params[params.configuration]['Mu_tracker_upstream']['TT_POSITIONS'])
nb_of_downstream_mu_plane = len(params.snd_params[params.configuration]['Mu_tracker_downstream']['TT_POSITIONS'])

parser = argparse.ArgumentParser()
parser.add_argument("step", help="First, run \"python Preprocessing_script.py Root2Pickle\". Then, if you want to display an event, run \"python Preprocessing_script.py Event_display\" ")
args = parser.parse_args()

# -----------------------------------------Produce the pickle files ------------------------------------------------------------------
if(args.step=="Root2Pickle"):
    print("Producing tt_cleared.pkl & y_cleared.pkl file by chunk in " + loc_of_pkl_file)

    # It is reading and analysing data by chunk instead of all at the time (memory leak problem)
    data_preprocessor = DataPreprocess(params)
    # create a directory where to store the data 
    os.system("mkdir -p {}".format(processed_file_path)) 
    for i in tqdm(range(0,n_steps)):  # tqdm: make your loops show a progress bar in terminal
        raw_chunk = data_preprocessor.open_shower_file(filename, start=i*step_size, stop=(i+1)*step_size) # opening data by chunk 
        outpath = processed_file_path + "/{}".format(i+begining_file)
        os.system("mkdir -p {}".format(outpath)) # create a directory where to store the pickle files of the ieme chunks
        data_preprocessor.clean_data_and_save(raw_chunk, outpath) # create the tt_cleared.pkl and y_cleared.pkl file in the directory

# ------------------------------------------Use this to display an event--------------------------------------------------------------
if(args.step=="Event_display"):
    print("Display an event:")

    reindex_TT_df, reindex_Mu_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)

    response_scifi = digitize_signal_scifi(reindex_TT_df.iloc[index], params=params, filters=nb_of_scifi_plane)
    response_upstream_mu = digitize_signal_upstream_mu(reindex_Mu_df.iloc[index], params=params, filters=nb_of_upstream_mu_plane)
    response_downstream_mu = digitize_signal_downstream_mu(reindex_Mu_df.iloc[index], params=params, filters=nb_of_downstream_mu_plane)
    #response.shape is an important parameter: it is the input size of our CNN
    print("Response Scifi tracker shape:",response_scifi.shape)
    print("Response Mu tracker shape:",response_upstream_mu.shape)
    print("Response Mu tracker shape:",response_downstream_mu.shape)

    plt.figure(figsize=(18,nb_of_scifi_plane))
    for i in range(nb_of_scifi_plane):
        plt.subplot(1,nb_of_scifi_plane,i+1)
        plt.imshow(response_scifi[i].astype("uint8") * 255, cmap='gray')
    plt.show()

    plt.figure(figsize=(18,nb_of_upstream_mu_plane))
    for i in range(nb_of_upstream_mu_plane):
        plt.subplot(1,nb_of_upstream_mu_plane,i+1)
        plt.imshow(response_upstream_mu[i].astype("uint8") * 255, cmap='gray')
    plt.show()

    plt.figure(figsize=(18,nb_of_downstream_mu_plane))
    for i in range(nb_of_downstream_mu_plane):
        plt.subplot(1,nb_of_downstream_mu_plane,i+1)
        plt.imshow(response_downstream_mu[i].astype("uint8") * 255, cmap='gray')
    plt.show()

# ------------------------------------------Use this to study n_hits_treshold--------------------------------------------------------------
if(args.step=="n_hits_treshold"):
    print("n_hits_treshold study:")

    # range of study of n_treshold
    n_hits_treshold_range = 15

    reindex_TT_df, reindex_Mu_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)

    #number of hits in SciFi
    TT_n_hits = reindex_TT_df.X.map(lambda x: len(x))
    #number of hits in muon planes
    Mu_n_hits = reindex_Mu_df.X.map(lambda x: len(x))
    #number of hits downstream mu tracker (3 planes)
    Mu_n_hits_downstream = reindex_Mu_df.Z.map(lambda x: x>147)
    n_hits_downstream = Mu_n_hits_downstream.map(lambda x: sum(x))
    #number of hits upstream mu tracker (5 planes)
    Mu_n_hits_upstream = reindex_Mu_df.Z.map(lambda x: x<147)
    n_hits_upstream = Mu_n_hits_upstream.map(lambda x: sum(x))

    TT_events_remaining = []
    Mu_events_remaining = []
    downstream_hit_events = []
    upstream_hit_events = []
    combined_events_remaining = []
    combined_TT_Mudown = []    

    for i in range(n_hits_treshold_range):
        TT_events_remaining.append(len(reindex_TT_df[TT_n_hits >= i]))
        Mu_events_remaining.append(len(reindex_Mu_df[(Mu_n_hits >= i)]))
        combined_events_remaining.append(len(reindex_TT_df[(Mu_n_hits >= i) | (TT_n_hits >= i)]))
        downstream_hit_events.append(len(n_hits_downstream[n_hits_downstream>= i]))
        upstream_hit_events.append(len(n_hits_upstream[n_hits_upstream>= i]))
        combined_TT_Mudown.append(len(n_hits_downstream[(n_hits_downstream >= i) | (TT_n_hits >= i)]))

    n_tot=file_size
    x = range(n_hits_treshold_range)

    plt.plot(x, [i/n_tot for i in TT_events_remaining], label='[hits_in_TT >= i]')
    plt.plot(x, [i/n_tot for i in Mu_events_remaining], label='[hits_in_Mu >= i]')
    plt.plot(x, [i/n_tot for i in downstream_hit_events], label='[hits_in_Mu_downstream >= i]')
    plt.plot(x, [i/n_tot for i in upstream_hit_events], label='[hits_in_Mu_upstream >= i]')
    plt.xlabel('n_hits_treshold')
    plt.ylabel('events remaining')
    plt.legend()
    plt.show()

    plt.plot(x, [i/n_tot for i in combined_events_remaining], label='[hits_in_TT >= i] or [hits_in_Mu >= i]')
    plt.plot(x, [i/n_tot for i in combined_TT_Mudown], label='[hits_in_TT >= i] or [hits_in_Mu_downstream >= i]')
    plt.xlabel('n_hits_treshold')
    plt.ylabel('events remaining')
    plt.legend()
    plt.show()

