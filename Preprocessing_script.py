# --------------------------------------------------------------IMPORT AND OPTIONS FOR PREPROCESSING ----------------------------------------------------------------------
# Import Class from utils.py & net.py file
from utils import DataPreprocess, Parameters, Parameters_reduced
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
#plt.ioff()

# ---------------------------------------------------- Barycenter fonction -------------------------------------------------------------------------------------------------

def compute_barycenter_of_event(event_data, params):

    #event_data = reindex_TT_df.iloc[index]

    nb_of_hit_per_plane = []
    z_plane = []
    barycenter = []
    nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])

    for i in range(nb_of_plane):
        nb_of_hit_per_plane.append(np.sum([(event_data["Z"]>= params.snd_params[params.configuration]["TT_POSITIONS"][i][0])
                                     & (event_data["Z"]<= params.snd_params[params.configuration]["TT_POSITIONS"][i][1])]))
        z_plane.append( (params.snd_params[params.configuration]["TT_POSITIONS"][i][0] + params.snd_params[params.configuration]["TT_POSITIONS"][i][1])*0.5 )

    # printing the first and the second max element in nb_of_hit_per_plane
    cp=list(nb_of_hit_per_plane)
    cp.sort()

    # If too many few events, all the barycenters are at the center of the planes
    if (cp[-2] <= 2) ^ (cp[-1] <= 2):
        if (cp[-1] <= 2):
            for i in range(nb_of_plane):
                barycenter.append([0,0,z_plane[i]])
        else:
            for i in range(nb_of_plane):
                barycenter.append([0,0,z_plane[i]])

            bool_max_plane = np.array((event_data["Z"]  >= params.snd_params[params.configuration]["TT_POSITIONS"][nb_of_hit_per_plane.index(cp[-1])][0])
                                    & (event_data["Z"] <= params.snd_params[params.configuration]["TT_POSITIONS"][nb_of_hit_per_plane.index(cp[-1])][1]))
            barycenter[nb_of_hit_per_plane.index(cp[-1])][0]=np.mean(event_data["X"][bool_max_plane])
            barycenter[nb_of_hit_per_plane.index(cp[-1])][1]=np.mean(event_data["Y"][bool_max_plane])

    else:

      if (cp[-2] == cp[-1]):
        index_max_nb_of_hit_per_plane = nb_of_hit_per_plane.index(cp[-1])
        nb_of_hit_per_plane[index_max_nb_of_hit_per_plane] +=1
        index_2max_nb_of_hit_per_plane= nb_of_hit_per_plane.index(cp[-2])
        nb_of_hit_per_plane[index_max_nb_of_hit_per_plane] -=1
      else:
        index_max_nb_of_hit_per_plane = nb_of_hit_per_plane.index(cp[-1])
        index_2max_nb_of_hit_per_plane= nb_of_hit_per_plane.index(cp[-2])

      #compute the barycenter of this 2 layers

      bool_max_plane = np.array((event_data["Z"]  >= params.snd_params[params.configuration]["TT_POSITIONS"][index_max_nb_of_hit_per_plane][0])
                   & (event_data["Z"] <= params.snd_params[params.configuration]["TT_POSITIONS"][index_max_nb_of_hit_per_plane][1]))

      bool_2max_plane = np.array((event_data["Z"]  >= params.snd_params[params.configuration]["TT_POSITIONS"][index_2max_nb_of_hit_per_plane][0])
                   & (event_data["Z"] <= params.snd_params[params.configuration]["TT_POSITIONS"][index_2max_nb_of_hit_per_plane][1]))

      barycenter_ref = []
      barycenter_ref.append([ np.mean(event_data["X"][bool_max_plane]),np.mean(event_data["Y"][bool_max_plane]),z_plane[index_max_nb_of_hit_per_plane] ])
      barycenter_ref.append([np.mean(event_data["X"][bool_2max_plane]) ,np.mean(event_data["Y"][bool_2max_plane]) ,z_plane[index_2max_nb_of_hit_per_plane] ])

      #compute the barycenter of the other one by interpolating a line between the first 2 barycenter computed
      for i in range(nb_of_plane):
          x_barycenter = barycenter_ref[0][0] +(barycenter_ref[1][0]-barycenter_ref[0][0])*(z_plane[i]-barycenter_ref[0][2])/(barycenter_ref[1][2]-barycenter_ref[0][2])
          y_barycenter = barycenter_ref[0][1] +(barycenter_ref[1][1]-barycenter_ref[0][1])*(z_plane[i]-barycenter_ref[0][2])/(barycenter_ref[1][2]-barycenter_ref[0][2])
          barycenter.append([ x_barycenter , y_barycenter ,z_plane[i] ])

    return barycenter

# ----------------------------------------------------  Reading the tt_cleared.pkl & y_cleared.pkl files by chunk ------------------------------------------------------

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

# ------------------------------------------ Reading the tt_cleared_reduced.pkl & y_cleared.pkl files by chunk ------------------------------------------------------------------

def reading_reduced_pkl_files(n_steps,processed_file_path):
    chunklist_TT_df_reduced  = []  # list of the TT_df file of each chunk
    chunklist_y_full = [] # list of the y_full file of each chunk
    
    #First 2 
    outpath = processed_file_path + "/{}".format(0)
    chunklist_TT_df_reduced.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
    outpath = processed_file_path + "/{}".format(1)
    chunklist_TT_df_reduced.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

    reindex_TT_df_reduced = pd.concat([chunklist_TT_df_reduced[0],chunklist_TT_df_reduced[1]],ignore_index=True)
    reindex_y_full = pd.concat([chunklist_y_full[0],chunklist_y_full[1]], ignore_index=True)

    for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
        outpath = processed_file_path + "/{}".format(i+2)
        chunklist_TT_df_reduced.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl"))) #add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
        reindex_TT_df_reduced = pd.concat([reindex_TT_df_reduced,chunklist_TT_df_reduced[i+2]], ignore_index=True)
        reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

    #reset empty space
    chunklist_TT_df_reduced = []
    chunklist_y_full = []
    
    return reindex_TT_df_reduced, reindex_y_full
# ---------------------------------------------------- Paramaters -----------------------------------------------------------------------------------------------------

# Here we choose the 4X0 geometry, which correspond to SND@LHC pilot run
# You need to change the X/Y half DIM in the .json file to 26.0 and 21.5
params = Parameters("4X0")

# Path to the raw Data root file and the pickle file
filename = "/home/debryas/DS5/DS5.root"
loc_of_pkl_file = "$HOME/DS5/ship_tt_processed_data_test"
processed_file_path = os.path.expandvars(loc_of_pkl_file)
name_of_angle_file = "results/Angle_histo.root"
name_of_red_dim_hist = "results/XY_histo.root"

# Usualy, Data file is too large to be read in one time, that the reason why we read it by "chunk" of step_size events
step_size = 100    # number of event in a chunk
file_size = 200  # number of events to be analyse. Maximum number for DS5.root is 200'000

n_steps = int(file_size / step_size) # number of chunks
nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])

parser = argparse.ArgumentParser()
parser.add_argument("step", help="As the code is a bit long, it is better to run it by step, to make sure everything works fine on the fly. First, run python Preprocessing_script.py step1, then Preprocessing_script.py step2, etc until stepX")
args = parser.parse_args()

# results of step4
reduced_dimension = []
reduced_dimension.append(7.986)
reduced_dimension.append(7.712)

params_reduced = Parameters_reduced("4X0")

# ----------------------------------------- PRODUCE THE tt_cleared.pkl & y_cleared.pkl IN ship_tt_processed_data/ FOLDER -------------------------------------------------
if(args.step=="step1"):
    print("Step1: producing tt_cleared.pkl & y_cleared.pkl file in " + loc_of_pkl_file)

    # It is reading and analysing data by chunk instead of all at the time (memory leak problem)
    print("\nProducing the tt_cleared.pkl & y_cleared.pkl files by chunk")
    data_preprocessor = DataPreprocess(params)
    os.system("mkdir -p {}".format(processed_file_path)) # create a directory ship_tt_processed_data where to store the data 
    for i in tqdm(range(0,n_steps)):  # tqdm: make your loops show a progress bar in terminal
        raw_chunk = data_preprocessor.open_shower_file(filename, start=i*step_size, stop=(i+1)*step_size) # opening data by chunk 
        outpath = processed_file_path + "/{}".format(i)
        os.system("mkdir -p {}".format(outpath)) # create a directory named 'i' where to store the data files of the 100 chunk
        data_preprocessor.clean_data_and_save(raw_chunk, outpath) # create the tt_cleared.pkl and y_cleared.pkl file in each of the directory

# ------------------------------------------ LOAD THE reindex_TT_df & reindex_y_full PD.DATAFRAME --------------------------------------------------------------------------
if(args.step=="step2"):
    
    print("Step2: Display an event and compute the barycenter of each planes")
    
    reindex_TT_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)

    #----------------------------------------- Ploting figure of the 6 component of TT_df --------------------------------------------------------------------------------------

    #index of the event you want to plot the signal
    index=7

    response = digitize_signal(reindex_TT_df.iloc[index], params=params, filters=nb_of_plane)
    print("Response shape:",response.shape) # gives (6,150,185) for resolution =700 and (6,525,645) for resolution =200 and (6, 75, 93) for resolution= 1400 // (6, 724, 865), res 1050 
    plt.figure(figsize=(18,nb_of_plane))
    for i in range(nb_of_plane):
        plt.subplot(1,nb_of_plane,i+1)
        plt.imshow(response[i].astype("uint8") * 255, cmap='gray')
    plt.show()

    # ----------------------------------------------------Compute the barycenter of an event----------------------------------------------------------------

    barycenter = compute_barycenter_of_event(reindex_TT_df.iloc[index], params)
    print(barycenter)

# --------------------------------------------------------Compute the angle of all the events----------------------------------------------------------------

if(args.step=="step3"):

    print("Step3: Compute the angle of all the events and save it in the file " + name_of_angle_file)

    reindex_TT_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)

    print("\nCompute the angle of all the events")
    from numpy.linalg import norm

    histRTheta = TH1F('histRTheta', 'Angle of electron', 100, -1, 11)
    histCTheta = TH1F('histCTheta', 'Computed angle of electron', 100, -1, 50)
    histDTheta = TH1F('histDTheta', 'Delta angle', 100, -11, 50)

    THETA = []
    y = []

    index=0
    droite_z = np.array([0,0,1])
    for h in tqdm(range(n_steps)):
    #for h in tqdm(range(10)):
        outpath = processed_file_path + "/{}".format(h)
        len_of_tt_cleared = len(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
        for i in range(len_of_tt_cleared):
            barycenter = compute_barycenter_of_event(reindex_TT_df.iloc[i+index], params)
            droite_event = np.array(barycenter[3])-np.array(barycenter[0])
            angle = np.arccos(np.dot(droite_event,droite_z)/norm(droite_z)/norm(droite_event))*180/np.pi
    #       THETA.append(angle)
    #       y.append(reindex_y_full["THETA"][index+i]*180/np.pi)
            histCTheta.Fill(angle)
            histRTheta.Fill(reindex_y_full["THETA"][index+i]*180/np.pi)
            histDTheta.Fill(angle-(reindex_y_full["THETA"][index+i]*180/np.pi))

        index = index+len_of_tt_cleared

    myfile = TFile(name_of_angle_file, 'RECREATE' )
    histRTheta.Write()
    histCTheta.Write()
    histDTheta.Write()
    myfile.Close()



# ---------------------------------------------------- Compute the reduced dimension that include 99% of the hits (reduced_dimension) -------------------------------------------

if(args.step=="step4"):
    print("Step4: Compute the reduced dimension that include 99% of the hits and save it in the file " + name_of_red_dim_hist)
    
    reindex_TT_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)

    histX = TH1F('histX', 'hist X', 100, -21, 21)
    histY = TH1F('histY', 'hist Y', 100, -20, 20)
    histX_bary = TH1F('histX_bary', 'hist X barycentered', 100, -21, 21)
    histY_bary = TH1F('histY_bary', 'hist Y barycentered', 100, -21, 21)

    for i in tqdm(range(len(reindex_TT_df))):
        barycenter = compute_barycenter_of_event(reindex_TT_df.iloc[i], params)
        for j in range(nb_of_plane):
            bool_plane = np.array( (reindex_TT_df.iloc[i]["Z"]  >= params.snd_params[params.configuration]["TT_POSITIONS"][j][0])
                       & (reindex_TT_df.iloc[i]["Z"] <= params.snd_params[params.configuration]["TT_POSITIONS"][j][1]) )
            x_pos_bary = reindex_TT_df.iloc[i]["X"][bool_plane]-barycenter[j][0]
            y_pos_bary = reindex_TT_df.iloc[i]["Y"][bool_plane]-barycenter[j][1]
            #To see the different with value unbarycentered (barycenter <=> (0,0) )       
            x_pos = reindex_TT_df.iloc[i]["X"][bool_plane]
            y_pos = reindex_TT_df.iloc[i]["Y"][bool_plane]
            for h in range(len(x_pos)):
                histX.Fill(x_pos[h])
                histY.Fill(y_pos[h])
                histX_bary.Fill(x_pos_bary[h])
                histY_bary.Fill(y_pos_bary[h])

    myfile = TFile( name_of_red_dim_hist, 'RECREATE' )
    histX.Write()
    histY.Write()
    histX_bary.Write()
    histY_bary.Write()
    myfile.Close()
    
    reduced_dimension = []
    reduced_dimension.append(3*histX_bary.GetStdDev())
    reduced_dimension.append(3*histY_bary.GetStdDev())

    print("Reduced dimension: ",reduced_dimension)
    print("\n Now you can enter those values in the reduced_dimension vector (line 170 of Preprocessing_script.py) and in the parameters_reduced.json file")


# ---------------------------------------------------------------------------------Compute the number of loose hit per events----------------------------------------------

if(args.step=="step5"):
    print("Step5: Compute the number of loose hit per events and plotting the events with at least 50\% of hits lost (noisy events)" )

    reindex_TT_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)
    
    number_of_it = 100
    ratio = []
    noisy_events_index = []
    
    # iterating over all events is very long so I decided to sample this analysis over a subsample of events (number_of_it)
    #for i in range(len(reindex_TT_df)):
    
    for i in tqdm(range(number_of_it)):
        barycenter = compute_barycenter_of_event(reindex_TT_df.iloc[i], params)
        hit = 0
        loose_hit = 0
        for j in range(nb_of_plane):
            bool_plane = np.array( (reindex_TT_df.iloc[i]["Z"]  >= params.snd_params[params.configuration]["TT_POSITIONS"][j][0])
                       & (reindex_TT_df.iloc[i]["Z"] <= params.snd_params[params.configuration]["TT_POSITIONS"][j][1]) )
            x_pos = reindex_TT_df.iloc[i]["X"][bool_plane]
            y_pos = reindex_TT_df.iloc[i]["Y"][bool_plane]
        
            if (len(x_pos) != len(y_pos)):
                print("x_pos != y_pos for event = ",i," and plane ",j)

            hit += len(x_pos)
        
            bool_X_plane = np.array(((x_pos-barycenter[j][0])  >= -reduced_dimension[0]) & ((x_pos-barycenter[j][0]) <= reduced_dimension[0])) 
            bool_Y_plane = np.array(((y_pos-barycenter[j][1])  >= -reduced_dimension[1]) & ((y_pos-barycenter[j][1]) <= reduced_dimension[1]))

            bool_XY_plane = bool_X_plane & bool_Y_plane
            loose_hit += (len(x_pos)-len(x_pos[bool_XY_plane]))
        
        if (100*loose_hit/hit)> 50:
            noisy_events_index.append(i)

        ratio.append(100*loose_hit/hit)
        
    print("Noisy enent: ", noisy_events_index)

    print("Mean ratio: ", np.mean(ratio))

    plt.figure()
    plt.plot(range(number_of_it),ratio)
    plt.title('Loose hit per events')
    plt.ylabel('ratio of loose hit [%]')
    plt.xlabel('event [ ]')
    plt.show()

    #Plotting the events  with (100*loose_hit/hit)>  50%

    for index in noisy_events_index:
        response = digitize_signal(reindex_TT_df.iloc[index], params=params, filters=nb_of_plane)
        plt.figure(figsize=(18,nb_of_plane))
        for i in range(nb_of_plane):
            plt.subplot(1,nb_of_plane,i+1)
            plt.imshow(response[i].astype("uint8") * 255, cmap='gray')

# --------------------------------------------------------PRODUCE THE tt_cleared_reduced.pkl IN ship_tt_processed_data/ FOLDER----------------------------

if(args.step=="step6"):

    print("Step6: Producing the tt_cleared_reduced.pkl files by chunk in " + loc_of_pkl_file)
    
    reindex_TT_df, reindex_y_full = reading_pkl_files(n_steps,processed_file_path)
    index=0
    for h in tqdm(range(0,n_steps)):
        List_vector = []
        outpath = processed_file_path + "/{}".format(h)
        len_of_tt_cleared = len(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
        for i in range(len_of_tt_cleared):
            barycenter = compute_barycenter_of_event(reindex_TT_df.iloc[i+index], params)
            bool_XY_plane= [False for j in range(len(reindex_TT_df.iloc[i+index]["Z"]))]
            X_position_bary = np.zeros(len(reindex_TT_df.iloc[i+index]['X']))
            Y_position_bary = np.zeros(len(reindex_TT_df.iloc[i+index]['Y']))
            for k in range(nb_of_plane):
                bool_plane = np.array( (reindex_TT_df.iloc[i+index]["Z"]  >= params.snd_params[params.configuration]["TT_POSITIONS"][k][0])
                                     & (reindex_TT_df.iloc[i+index]["Z"]  <= params.snd_params[params.configuration]["TT_POSITIONS"][k][1]) )

                bool_X = np.array( ((reindex_TT_df.iloc[i+index]["X"]-barycenter[k][0])  >= -reduced_dimension[0])
                                 & ((reindex_TT_df.iloc[i+index]["X"]-barycenter[k][0])  <=  reduced_dimension[0]) )

                bool_Y = np.array( ((reindex_TT_df.iloc[i+index]["Y"]-barycenter[k][1])  >= -reduced_dimension[1])
                                 & ((reindex_TT_df.iloc[i+index]["Y"]-barycenter[k][1])  <=  reduced_dimension[1]) )

                bool_XY_plane= (bool_XY_plane) ^ (bool_X & bool_Y & bool_plane)

                X_position_bary += (reindex_TT_df.iloc[i+index]['X']-barycenter[k][0])*bool_plane

                Y_position_bary += (reindex_TT_df.iloc[i+index]['Y']-barycenter[k][1])*bool_plane

            TT_resp = {
                'PX': reindex_TT_df.iloc[i+index]['PX'][bool_XY_plane],
                'PY': reindex_TT_df.iloc[i+index]['PY'][bool_XY_plane],
                'PZ': reindex_TT_df.iloc[i+index]['PZ'][bool_XY_plane],

                'X': X_position_bary[bool_XY_plane],
                'Y': Y_position_bary[bool_XY_plane],
                'Z': reindex_TT_df.iloc[i+index]['Z'][bool_XY_plane],

                'Time': reindex_TT_df.iloc[i+index]['Time'][bool_XY_plane],
                'PdgCode': reindex_TT_df.iloc[i+index]['PdgCode'][bool_XY_plane],
                'AssociatedMCParticle': reindex_TT_df.iloc[i+index]['AssociatedMCParticle'][bool_XY_plane],
                'ELoss': reindex_TT_df.iloc[i+index]['ELoss'][bool_XY_plane]
            }
            List_vector.append(TT_resp)

        tt_cleared_reduced = pd.DataFrame(List_vector)
        tt_cleared_reduced.to_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl"))
        index = index+len_of_tt_cleared

# ----------------------------------------------------------Compute the dimension of the reindex_TT_df_reduced image for the net.py-------------------------------------

if(args.step=="step7"):

    print("Step7: Compute the dimension of the reindex_TT_df_reduced images for net.py, and display an event (you can modify the display event by changing the index number)")
    reindex_TT_df_reduced, reindex_y_full = reading_reduced_pkl_files(n_steps,processed_file_path)

    #  !!!!!!!!!!!!!!!!!! CHANGE THE PARAMETER FILE FOR X AND Y DIMENSION  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #index of the event you want to plot the signal
    index=7
    response = digitize_signal(reindex_TT_df_reduced.iloc[index], params=params_reduced, filters=nb_of_plane)
    print("Response shape:",response.shape) # gives (6,150,185) for resolution =700 and (6,525,645) for resolution =200 and (6, 75, 93) for resolution= 1400 // (6, 724, 865), res 1050 
    plt.figure(figsize=(18,nb_of_plane))
    for i in range(nb_of_plane):
        plt.subplot(1,nb_of_plane,i+1)
        plt.imshow(response[i].astype("uint8") * 255, cmap='gray')
    plt.show()
