# --------------------------------------------------------------IMPORT AND OPTIONS FOR PREPROCESSING ----------------------------------------------------------------------
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
# Turn off interactive plotting: for long run it makes screwing up everything
plt.ioff()

# ----------------------------------------------------  Reading the tt_cleared.pkl & y_cleared.pkl files by chunk -------------------------------------

def reading_pkl_files(n_steps,processed_file_path):
    chunklist_TT_df = []  # list of the TT_df file of each chunk
    chunklist_y_full = [] # list of the y_full file of each chunk

    # It is reading and analysing data by chunk instead of all at the time (memory leak problem)
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
        chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl"))) # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
        reindex_TT_df = pd.concat([reindex_TT_df,chunklist_TT_df[i+2]], ignore_index=True)
        reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

    #reset empty space
    chunklist_TT_df = []
    chunklist_y_full = []

    return reindex_TT_df, reindex_y_full

# ---------------------------------------------------- Paramaters -----------------------------------------------------------------------------------------------------

params = Parameters("SNDatLHC")

# Path to the raw Data root file and the pickle file
filename = "$HOME/Desktop/data/heavy_data/nue_CCDIS_0to200kevents.root"
loc_of_pkl_file = "$HOME/Desktop/data/processed_data/CCDIS"
processed_file_path = os.path.expandvars(loc_of_pkl_file)

# Usualy, Data file is too large to be read in one time, that the reason why we read it by "chunk" of step_size events
step_size = 1000    # number of event in a chunk
file_size = 200000  # number of events to be analyse. Maximum number for DS5.root is 200'000

n_steps = int(file_size / step_size) # number of chunks
nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])
begining_file = 0

parser = argparse.ArgumentParser()
parser.add_argument("step", help="As the code is a bit long, it is better to run it by step, to make sure everything works fine on the fly. First, run python Preprocessing_script.py step1, then Preprocessing_script.py step2, etc until stepX")
args = parser.parse_args()

# ----------------------------------------- PRODUCE THE tt_cleared.pkl & y_cleared.pkl IN ship_tt_processed_data/ FOLDER -------------------------------------------------
if(args.step=="step1"):
    print("Step1: producing tt_cleared.pkl & y_cleared.pkl file in " + loc_of_pkl_file)

    # It is reading and analysing data by chunk instead of all at the time (memory leak problem)
    print("\nProducing the tt_cleared.pkl & y_cleared.pkl files by chunk")
    data_preprocessor = DataPreprocess(params)
    os.system("mkdir -p {}".format(processed_file_path)) # create a directory ship_tt_processed_data where to store the data 
    for i in tqdm(range(0,n_steps)):  # tqdm: make your loops show a progress bar in terminal
        raw_chunk = data_preprocessor.open_shower_file(filename, start=i*step_size, stop=(i+1)*step_size) # opening data by chunk 
        outpath = processed_file_path + "/{}".format(i+begining_file)
        os.system("mkdir -p {}".format(outpath)) # create a directory named 'i+begining_file' where to store the data files of the 100 chunk
        data_preprocessor.clean_data_and_save(raw_chunk, outpath) # create the tt_cleared.pkl and y_cleared.pkl file in each of the directory

# ------------------------------------------ LOAD THE reindex_TT_df & reindex_y_full PD.DATAFRAME --------------------------------------------------------------------------
if(args.step=="step2"):
    
    print("Step2: Display an event of each planes")
    
    reindex_TT_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)

    #----------------------------------------- Ploting figure of the 6 component of TT_df --------------------------------------------------------------------------------------

    #index of the event you want to plot the signal
    index=91

    response = digitize_signal(reindex_TT_df.iloc[index], params=params, filters=nb_of_plane)
    print("Response shape:",response.shape) # gives (6,150,185) for resolution =700 and (6,525,645) for resolution =200 and (6, 75, 93) for resolution= 1400 // (6, 724, 865), res 1050 
    plt.figure(figsize=(18,nb_of_plane))
    for i in range(nb_of_plane):
        plt.subplot(1,nb_of_plane,i+1)
        plt.imshow(response[i].astype("uint8") * 255, cmap='gray')
    plt.show()



