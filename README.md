# EM shower reconstruction with the SciFi at SND@LHC
ML code for energy reconstruction of neutrino of the SND@LHC geometry

## Instructions by Paul 

[in progress]

### To do:

0) ship_tt.yml : Before anything, you need to create the needed environement with the folowing comand "conda env create -f ship_tt.yml" (to do once). Then, each time you log in, you need to activate the environment with the command "conda activate ship_tt".

1) parameters/parameters.py: In this file, you should write the geometry parameters of your SND@LHC detector. Then run "python parameters.py" to create the parameters.json file inside the parameters/ folder, needed for the code.

2) Preprocessing_script.py: This file is use to transform the raw data file (root file) to processed data (pickle file), in which only the needed informations are stored. It is reading and analysing data by chunk instead of all at the same time because of memory leak problem. Which mean from a single root file, corresponding pickle files will be stored in n different folders (from 0 to n), and in each of these folders, we will find 2 pickles files:
tt_cleared.pkl: Information of the tracker (position of the hits in the scifi planes, ...)
y_cleared.pkl:  Information on the event (energy of the neutrino, Elastic or inelastic event, ...)

First, you need to run "python Preprocessing_script.py Root2Pickle". Then, if you want to display an event, run "python Preprocessing_script.py Event_display" to show what an event looks like (it also display the images dimension: the input of our CNN).

3) Paul_run_script.py: Main script, where the ML magic happens. It train the CNN define in class "SNDNet" (net.py), and it save the network and the prediction every 10 epoch. You can run it with "python Paul_run_script.py".


In the other files, only functions are written:

-utils.py: useful function, mainly for the preprocessing part.

-net.py: Class and function for the CNN. Class SNDNet describe the geometry of the network, you can play with it.

-coord_conv.py: useful functions for the implementation of the CoordConvmethod describe in the folowing paper: https://arxiv.org/pdf/1807.03247.pdf

### usual mistakes and how to handle them

## Useful links
 - FairShip: https://github.com/ShipSoft/FairShip
 - Intro on python, bash and git: https://hsf-training.github.io/analysis-essentials/
 - Stanford course on neural networks: http://cs231n.stanford.edu/2019/
 - Paul's thesis: https://lphe.epfl.ch/publications/theses/Master_Thesis_Paul_De_Bryas.pdf
 - SND@LHC proposal: https://cds.cern.ch/record/2709550/files/2002.08722.pdf

## Associated projects
 - SND@LHC, later (if approved) SHiP: event reconstruction in real time using the SCiFi as a sampling calorimeter. The SND is a detector made of emulsion bricks interleaved with SciFi planes, followed by a muon detector. It will be placed in the TI18 cavern, pointing at the ATLAS interaction region. It will detect neutrals (neutrinos + long-lived) produced at IP1. The same detector, in a bigger version, will constitute SHiP's SND (Scattering and Neutrino Detector). Emulsions are great at reconstructing neutrino interactions, with the "small" problem of being an integrating detector... That's why they are interleaved with SciFi planes, that will allow disentangling pile-up in the bricks as well as add time stamps. However, there's a certain level of analysis that can be done in real time using only SciFi information. The first part, EM shower reconstruction, has already been started, achieving energy reconstruction with a resolution of 5% at 100 GeV using a NN trained with electron events. But there's much more in the todo list: for example, gaining the ability of reconstructing lone tracks accompanying the EM shower. This helps in disentangling inelastic scatterings from v+e->v+e or DM+e->DM+e events.
 - [same introduction as above]: "optimising" / finalising the shower reconstruction algorithm: we have totally ignored the ghost hits problem, for example (the scifi planes will measure x/y with alternating layers), and we haven't studied what resolution on the shower energy and direction is needed to distinguish neutrino elastic scattering from dark matter scattering based on the different event kinematics due to nonzero dark matter mass, for example.
