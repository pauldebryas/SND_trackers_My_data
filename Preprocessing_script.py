#Preprocessing_script.py

# This file is use to transform the raw data file (root file) to processed data (pickle file), in which only the needed informations are stored.

# -----------------------------------IMPORT AND OPTIONS FOR PREPROCESSING-------------------------------------------------------------
# Import Class from utils.py & net.py file
from utils import DataPreprocess, Parameters
from net import digitize_signal

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
def reading_pkl_files_scifi_only(n_steps,processed_file_path):
    chunklist_TT_df = []  # list of the TT_df file of each chunk
    chunklist_y_full = [] # list of the y_full file of each chunk

    # It is reading and analysing data by chunk instead of all at the same time (memory leak problem)
    #First 2 
    outpath = processed_file_path + "/{}".format(0)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
    outpath = processed_file_path + "/{}".format(1)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

    reindex_TT_df = pd.concat([chunklist_TT_df[0],chunklist_TT_df[1]],ignore_index=True)
    reindex_y_full = pd.concat([chunklist_y_full[0],chunklist_y_full[1]], ignore_index=True)

    for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
        outpath = processed_file_path + "/{}".format(i+2)
        
        # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
        chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl"))) 
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) 
        
        reindex_TT_df = pd.concat([reindex_TT_df,chunklist_TT_df[i+2]], ignore_index=True)
        reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

    #reset empty space
    chunklist_TT_df = []
    chunklist_y_full = []

    return reindex_TT_df, reindex_y_full

# includes both scifi and muon processing
def reading_pkl_files(n_steps,processed_file_path):
    chunklist_TT_df = []  # list of the TT_df file of each chunk (scifi planes)
    chunklist_Mu_df = []  # list of the TT_df file of each chunk (muon planes)
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
        
        # add all the .pkl files read_pickle and add to the chunklist list
        chunklist_TT_df .append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl"))) 
        chunklist_Mu_df .append(pd.read_pickle(os.path.join(outpath, "mu_cleared.pkl")))
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath,  "y_cleared.pkl"))) 
        
        reindex_TT_df  = pd.concat([reindex_TT_df,  chunklist_TT_df [i+2]], ignore_index=True)
        reindex_Mu_df  = pd.concat([reindex_Mu_df,  chunklist_Mu_df [i+2]], ignore_index=True)
        reindex_y_full = pd.concat([reindex_y_full, chunklist_y_full[i+2]], ignore_index=True)

    #reset empty space
    chunklist_TT_df  = []
    chunklist_Mu_df  = []
    chunklist_y_full = []

    return reindex_TT_df, reindex_Mu_df, reindex_y_full

# ----------------------------------------------------Paramaters----------------------------------------------------------------------
# parameter of the geometry configuration
params = Parameters("SNDatLHC") 

# Path to the raw Data root file and the localisation of where to store the pickle file
filename = "$HOME/Desktop/data/heavy_data/nue_CCDIS_0to200kevents.root"
loc_of_pkl_file = "$HOME/Desktop/data/processed_data/CCDIS"
processed_file_path = os.path.expandvars(loc_of_pkl_file)

# Usualy, Data file is too large to be read in one time, that the reason why we read it by "chunk" of step_size events
# number of event in a chunk
step_size = 1000
# number of events to be analyse. Warning: Maximum number should not exceed the maximum number of events in the root file it must be a multiple of step_size
file_size = 200000
#file number to begging with. Useful when the raw data file is split in multiple file. Default value must be 0.
begining_file = 0

#index of the event you want to plot the signal
index=91 # Warning: this should not exceed the maximum number of events proceed. 
# Be careful, there is less events proceed than event in the root file (events with less than 5 hits are not proceed)

# ----------------------------------------------------Global variables----------------------------------------------------------------
n_steps = int(file_size / step_size) 
nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])
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

    reindex_TT_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)

    response = digitize_signal(reindex_TT_df.iloc[index], params=params, filters=nb_of_plane)
    #response.shape is an important parameter: it is the input size of our CNN
    print("Response shape:",response.shape) 
    plt.figure(figsize=(18,nb_of_plane))
    for i in range(nb_of_plane):
        plt.subplot(1,nb_of_plane,i+1)
        plt.imshow(response[i].astype("uint8") * 255, cmap='gray')
    plt.show()
