import numpy as np
import pandas as pd
import os


PART_TYPE_ARR = ['nuel', 'numu', 'nutau']


def read_chunklist(proc_file_path, step_size, file_size):
    # step_size = size of a chunk
    # file_size = size of the CCDIS BigFile

    n_steps = int(file_size / step_size) # number of chunks

    chunklist_scifi = [] # list of the TT_df file of each chunk
    chunklist_muons = [] # list of the TT_df file of each chunk
    chunklist_energ = [] # list of the y_full file of each chunk
    
    # first 2 
    for i in range(2):
        outpath = proc_file_path + "/{}".format(i)
        chunklist_scifi.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
        chunklist_muons.append(pd.read_pickle(os.path.join(outpath, "mu_cleared.pkl")))
        chunklist_energ.append(pd.read_pickle(os.path.join(outpath,  "y_cleared.pkl")))

    scifi_df = pd.concat([chunklist_scifi[0],
                          chunklist_scifi[1]], ignore_index=True)
    muons_df = pd.concat([chunklist_muons[0],
                          chunklist_muons[1]], ignore_index=True)
    energ_df = pd.concat([chunklist_energ[0],
                          chunklist_energ[1]], ignore_index=True)

    print("Before Reduction (file " + proc_file_path + "):")

    # other n-2
    for i in range(n_steps-2):
        outpath = proc_file_path + "/{}".format(i+2)
        
        # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        chunklist_scifi.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
        
        # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        chunklist_muons.append(pd.read_pickle(os.path.join(outpath, "mu_cleared.pkl")))

        # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
        chunklist_energ.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
        
        scifi_df = pd.concat([scifi_df, chunklist_scifi[i+2]], ignore_index=True)
        muons_df = pd.concat([muons_df, chunklist_muons[i+2]], ignore_index=True)
        energ_df = pd.concat([energ_df, chunklist_energ[i+2]], ignore_index=True)
    
    print("  TT_df  : " + str(len(scifi_df)))
    print("  MU_df  : " + str(len(muons_df)))
    print("  y_full : " + str(len(energ_df)))

    return scifi_df, muons_df, energ_df


def load_dataframes(params, paths_dict, step_size, files_num):
    full_paths = dict()

    for part_type in PART_TYPE_ARR:
        full_paths[part_type] = os.path.expandvars(paths_dict[part_type])
    
    # --- LOAD THE reindex_TT_df & reindex_y_full PD.DATAFRAME ---

    # It is reading and analysing data by chunk instead of all at the time (memory leak problem)
    print("\nReading the tt_cleared.pkl & y_cleared.pkl files by chunk of CCDIS and NueEElastic")
    
    scifi_arr = dict()
    mu_arr    = dict()
    en_arr    = dict()
    
    # step_size = events in one file
    # files_num = number of files
    file_size = files_num * step_size  # total number of events
    
    
    for part_type in PART_TYPE_ARR:
        scifi_arr[part_type], mu_arr[part_type], en_arr[part_type] = read_chunklist(full_paths[part_type], 
                                                                                    step_size, file_size)
        
    return scifi_arr, mu_arr, en_arr

# selecting events to ensure equal number of events for all particle types
def balance_events_num(scifi_arr, mu_arr, en_arr):
    event_limit = min(len(en_arr['nuel']),
                      len(en_arr['numu']),
                      len(en_arr['nutau']))

    for part_type in ['nuel', 'numu', 'nutau']:        
        scifi_arr[part_type] = scifi_arr[part_type][:event_limit]
        mu_arr   [part_type] = mu_arr   [part_type][:event_limit]
        en_arr   [part_type] = en_arr   [part_type][:event_limit]
    
    return scifi_arr, mu_arr, en_arr


def merge_events_arrays(scifi_arr, mu_arr, en_arr):
    # Merging CCDIS and NueEElastic in a single array
    combined_scifi_arr  = pd.concat([scifi_arr ['nuel'], 
                                     scifi_arr ['numu'],
                                     scifi_arr ['nutau']], ignore_index=True, sort=False)

    combined_mu_arr  = pd.concat([mu_arr ['nuel'], 
                                  mu_arr ['numu'],
                                  mu_arr ['nutau']], ignore_index=True, sort=False)

    combined_en_arr  = pd.concat([en_arr ['nuel'], 
                                  en_arr ['numu'],
                                  en_arr ['nutau']], ignore_index=True, sort=False)
    
    print("After Reduction  :\n")

    for part_type in ['nuel', 'numu', 'nutau']:
        print("Particle type: " + part_type)
        print("  scifi_arr : " + str(len(scifi_arr[part_type])))
        print("  mu_arr    : " + str(len(mu_arr[part_type])))
        print("  en_arr    : " + str(len(en_arr[part_type])))

    print()
    print("combined_scifi_arr : " + str(len(combined_scifi_arr)))
    print("combined_mu_arr: " + str(len(combined_mu_arr)))
    print("combined_en_arr: " + str(len(combined_en_arr)))

    return combined_scifi_arr, combined_mu_arr, combined_en_arr


def normalise_target_energy(en_arr, norm = 1. / 4000):
    # True value of NRJ for each true Nue event
    en_arr[["E"]] *= norm
    return en_arr


def read_pickled_df(detector_params, paths_dict, step_size, file_size):
    scifi_arr, mu_arr, en_arr = load_dataframes(detector_params, paths_dict, step_size, file_size)
    
    # not required
    # scifi_arr, mu_arr, en_arr = balance_events_num(scifi_arr, mu_arr, en_arr)
    scifi_arr, mu_arr, en_arr = merge_events_arrays(scifi_arr, mu_arr, en_arr)
    
    en_arr = normalise_target_energy(en_arr)
    
    return scifi_arr, mu_arr, en_arr