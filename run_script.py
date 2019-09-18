from utils import DataPreprocess, Parameters
import torch
try:
    assert(torch.cuda.is_available())
except:
    raise Exception("CUDA is not available")

n_devices = torch.cuda.device_count()
print("\nWelcome!\n\nCUDA devices available:\n")
for i in range(n_devices):
    print("\t{}\twith CUDA capability {}".format(torch.cuda.get_device_name(device=i), torch.cuda.get_device_capability(device=i)))
print("\n")
    
from net import SNDNet, MyDataset, digitize_signal
device = torch.device("cuda", 0)
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
import os
import gc

params = Parameters("9X0")
data_preprocessor = DataPreprocess(params)
filename = "/data/ship_tt/9X0_500k.root"

print("\nReading data now. Be patient...")
processed_file_path = os.path.expandvars("$HOME/ship_tt_processed_data")
os.system("mkdir -p {}".format(processed_file_path))
step_size = 5000
file_size = 500000
n_steps = int(file_size / step_size)
chunks = []
for i in tqdm(range(n_steps)):
    gc.collect()
    raw_chunk = data_preprocessor.open_shower_file(filename, start=i*step_size, stop=(i+1)*step_size)
    outpath = processed_file_path + "/{}".format(i)
    os.system("mkdir -p {}".format(outpath))
    data_preprocessor.clean_data_and_save(raw_chunk, outpath)

#print("\nMerging chunks...")
#showers_root = np.concatenate(chunks)

# print("\nData read. Cleaning and saving sample...")
#data_preprocessor.clean_data_and_save(showers_root, processed_file_path)
#TT_df = pd.read_pickle(os.path.join(processed_file_path, "tt_cleared.pkl"))
#y_full = pd.read_pickle(os.path.join(processed_file_path, "y_cleared.pkl"))

#print("\nDigitizing signal...")
#index=0
#response = digitize_signal(TT_df.iloc[index], params=params, filters=6)
#print(response.shape)
