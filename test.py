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

# ---------------------------------------------------- Paramaters -----------------------------------------------------------------------------------------------------

params = Parameters("SNDatLHC")

# Path to the raw Data root file and the pickle file
filename = "$HOME/Desktop/data/heavy_data/nue_NuEElastic_0to200kevents.root"
loc_of_pkl_file = "$HOME/Desktop/data/processed_data"
processed_file_path = os.path.expandvars(loc_of_pkl_file)

# Usualy, Data file is too large to be read in one time, that the reason why we read it by "chunk" of step_size events
step_size = 1000    # number of event in a chunk
file_size = 1000  # number of events to be analyse. Maximum number for DS5.root is 200'000

n_steps = int(file_size / step_size) # number of chunks
nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])
begining_file = 199

parser = argparse.ArgumentParser()
parser.add_argument("step", help="As the code is a bit long, it is better to run it by step, to make sure everything works fine on the fly. First, run python Preprocessing_script.py step1, then Preprocessing_script.py step2, etc until stepX")
args = parser.parse_args()

# ----------------------------------------- PRODUCE THE tt_cleared.pkl & y_cleared.pkl IN ship_tt_processed_data/ FOLDER -------------------------------------------------
if(args.step=="step1"):
    print("Step1: producing tt_cleared.pkl & y_cleared.pkl file in " + loc_of_pkl_file)

    # It is reading and analysing data by chunk instead of all at the time (memory leak problem)
    print("\nProducing the tt_cleared.pkl & y_cleared.pkl files by chunk")
    data_preprocessor = DataPreprocess(params)
    for i in tqdm(range(0,n_steps)):  # tqdm: make your loops show a progress bar in terminal
        raw_chunk = data_preprocessor.open_shower_file(filename, start=199000, stop=200000) # opening data by chunk 
        outpath = processed_file_path + "/{}".format(199)
        os.system("mkdir -p {}".format(outpath)) # create a directory named 'i+begining_file' where to store the data files of the 100 chunk
        data_preprocessor.clean_data_and_save(raw_chunk, outpath) # create the tt_cleared.pkl and y_cleared.pkl file in each of the directory
