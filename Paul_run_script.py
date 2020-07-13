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
processed_file_path = os.path.expandvars("$HOME/DS5/ship_tt_processed_data") #!!!!!!!!!!!!!!!!!!!!!CHANGE THE PATH !!!!!!!!!!!!!!!!
step_size = 5000    # size of a chunk
file_size = 180000  # size of the BigFile.root file
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

#----------------------------------------- Ploting figure of the 6 component of TT_df
'''
index=2
response = digitize_signal(reindex_TT_df.iloc[index], params=params, filters=nb_of_plane)
print("Response shape:",response.shape) # gives  (6, 250, 298) unreduced and (6, 132, 154) for reduced data --> change the network 
number_of_paramater = response.shape[0]
plt.figure(figsize=(18,number_of_paramater))
for i in range(number_of_paramater):
    plt.subplot(1,number_of_paramater,i+1)
    plt.imshow(response[i].astype("uint8") * 255, cmap='gray')
'''

# True value of NRJ/dist for each true electron event
#y = reindex_y_full[["E", "Z","THETA"]]
y = reindex_y_full[["E"]]
NORM = 1. / 100
y["E"] *= NORM
#y["Z"] *= -1
#y["THETA"] *= (180/np.pi)


# reset to empty space
reindex_y_full = []

# Spliting
print("\nSplitting the data into a training and a testing sample")

indeces = np.arange(len(reindex_TT_df))
train_indeces, test_indeces, _, _ = train_test_split(indeces, indeces, train_size=0.9, random_state=1543)

#batch_size = 512
batch_size = 150

train_dataset = MyDataset(reindex_TT_df, y, params, train_indeces, n_filters=nb_of_plane)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = MyDataset(reindex_TT_df, y, params, test_indeces, n_filters=nb_of_plane)
test_batch_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# reset to empty space
reindex_TT_df=[]

# Saving the true Energy for the test sample
TrueE_test=y["E"][test_indeces]
np.save("TrueE_test.npy",TrueE_test)

# Creating the network
net = SNDNet(n_input_filters=nb_of_plane).to(device)

# Loose rate, num epoch and weight decay parameters of our network backprop actions
lr = 1e-3
opt = torch.optim.Adam(net.model.parameters(), lr=lr, weight_decay=0.01)
num_epochs = 40

train_loss = []
val_accuracy_1 = []
val_accuracy_2 = []

# Create a directory where to store the 9X0 files
os.system("mkdir 9X0_file")


#Training
print("\nNow Trainig the network:")
# Create a .txt file where we will store some info for graphs
f=open("NN_performance.txt","a")
f.write("Epoch/Time it took (s)/Loss/Validation energy (%)/Validation distance (%)\n")
f.close()

class Logger(object):
    def __init__(self):
        pass

    def plot_losses(self, epoch, num_epochs, start_time):
        # Print and save in NN_performance.txt the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_loss[-1]))
        print("  validation Energy:\t\t{:.4f} %".format(val_accuracy_1[-1]))
        #print("  validation distance:\t\t{:.4f} %".format(val_accuracy_2[-1]))

        f=open("NN_performance.txt","a")
        f.write("{};{:.3f};".format(epoch + 1, time.time() - start_time))
        f.write("\t{:.6f};".format(train_loss[-1]))
        f.write("\t\t{:.4f}\n".format(val_accuracy_1[-1]))
        #f.write("\t\t{:.4f}\n".format(val_accuracy_2[-1]))
        f.close()


def run_training(lr, num_epochs, opt):
    try:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            start_time = time.time()
            net.model.train(True)
            epoch_loss = 0
            for X_batch, y_batch in tqdm(train_batch_gen, total = int(len(train_indeces) / batch_size)):
#            for X_batch, y_batch in train_batch_gen:
            # train on batch
                loss = net.compute_loss(X_batch, y_batch)
                loss.backward()
                opt.step()
                opt.zero_grad()
                epoch_loss += loss.item()

            train_loss.append(epoch_loss / (len(train_indeces) // batch_size + 1))

            y_score = []
            with torch.no_grad():
                for (X_batch, y_batch) in tqdm(test_batch_gen, total = int(len(test_indeces) / batch_size)):
                    logits = net.predict(X_batch)
                    y_pred = logits.cpu().detach().numpy()
                    y_score.extend(y_pred)

            y_score = mean_squared_error(y.iloc[test_indeces], np.asarray(y_score), multioutput='raw_values')
            val_accuracy_1.append(y_score[0])
            #val_accuracy_2.append(y_score[1])    

            # Visualize
            display.clear_output(wait=True)
            logger.plot_losses(epoch, num_epochs, start_time)

            #Saving network for each 10 epoch
            if (epoch + 1) % 10 == 0:
                with open("9X0_file/" + str(epoch) + "_9X0_coordconv.pt", 'wb') as f:
                    torch.save(net, f)       
                lr = lr / 2
                opt = torch.optim.Adam(net.model.parameters(), lr=lr)
    except KeyboardInterrupt:
        pass

logger = Logger()
run_training(lr, num_epochs, opt)

# Saving the prediction at each epoch

# Create a directory where to store the prediction files
os.system("mkdir PredE_file")

for i in [9,19,29,39]:
    net = torch.load("9X0_file/" + str(i) + "_9X0_coordconv.pt")
    preds = []
    with torch.no_grad():
        for (X_batch, y_batch) in test_batch_gen:
            preds.append(net.predict(X_batch))
    ans = np.concatenate([p.detach().cpu().numpy() for p in preds])
    np.save("PredE_file/" + str(i) + "_PredE_test.npy",ans[:, 0])
    print("Save Prediction for epoch "+ str(i))


