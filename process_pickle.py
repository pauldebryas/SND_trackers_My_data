import numpy as np
import pandas as pd
import os

def read_chunklist(proc_file_path, step_size, file_size):
    # step_size = size of a chunk
    # file_size = size of the CCDIS BigFile

    n_steps = int(file_size / step_size) # number of chunks

    chunklist_TT_df  = [] # list of the TT_df file of each chunk
    chunklist_y_full = [] # list of the y_full file of each chunk
    
    # first 2 
    for i in range(2):
        outpath = proc_file_path + "/{}".format(i)
        chunklist_TT_df .append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

    reindex_TT_df  = pd.concat([chunklist_TT_df[0],
                                chunklist_TT_df[1]],  ignore_index=True)
    reindex_y_full = pd.concat([chunklist_y_full[0],
                                chunklist_y_full[1]], ignore_index=True)

    print("Before Reduction (file " + proc_file_path + "):")

    # other n-2
    for i in range(n_steps-2):
        outpath = proc_file_path + "/{}".format(i+2)
        
        # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
        chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared.pkl")))
        
        # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
        chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
        
        reindex_TT_df  = pd.concat([reindex_TT_df,  
                                    chunklist_TT_df[i+2]], ignore_index=True)
        reindex_y_full = pd.concat([reindex_y_full, 
                                    chunklist_y_full[i+2]], ignore_index=True)
    
    print("  TT_df  inelastic: " + str(len(reindex_TT_df)))
    print("  y_full inelastic: " + str(len(reindex_y_full)))

    return reindex_TT_df, reindex_y_full


def load_dataframes(params, path_nuel, path_numu, path_nutau):
    
    # Here we choose the geometry with 9 time the radiation length
    proc_path_nuel  = os.path.expandvars(path_nuel)
    proc_path_numu  = os.path.expandvars(path_numu)
    proc_path_nutau = os.path.expandvars(path_nutau)
    
    # --- LOAD THE reindex_TT_df & reindex_y_full PD.DATAFRAME ---

    # It is reading and analysing data by chunk instead of all at the time (memory leak problem)
    print("\nReading the tt_cleared.pkl & y_cleared.pkl files by chunk of CCDIS and NueEElastic")
    
    reidx_TT_df  = dict()
    reidx_y_full = dict()
    
    step_size = 1000                   # events in one file
    files_num = 10                     # number of files
    file_size = files_num * step_size  # total number of events
    
    reidx_TT_df['nuel'],  reidx_y_full['nuel']  = read_chunklist(proc_path_nuel,  step_size, file_size)
    reidx_TT_df['numu'],  reidx_y_full['numu']  = read_chunklist(proc_path_numu,  step_size, file_size)
    reidx_TT_df['nutau'], reidx_y_full['nutau'] = read_chunklist(proc_path_nutau, step_size, file_size)
    
    return reidx_TT_df, reidx_y_full


def balance_events_num(reindex_TT_df, reindex_y_full):
    # Selecting events to ensure equal number of elastic and inelastic events
    event_limit = min(len(reindex_y_full['nuel']),
                      len(reindex_y_full['numu']),
                      len(reindex_y_full['nutau']))

    for part_type in ['nuel', 'numu', 'nutau']:
        remove = int(len(reindex_TT_df[part_type]) - event_limit) + 1
        reindex_TT_df [part_type] = reindex_TT_df [part_type][:-remove]
        reindex_y_full[part_type] = reindex_y_full[part_type][:-remove]
    
    return reindex_TT_df, reindex_y_full


def merge_events_arrays(reindex_TT_df, reindex_y_full):
    # Merging CCDIS and NueEElastic in a single array
    combined_TT_df  = pd.concat([reindex_TT_df ['nuel'], 
                                 reindex_TT_df ['numu'],
                                 reindex_TT_df ['nutau']], ignore_index=True, sort=False)
    combined_y_full = pd.concat([reindex_y_full['nuel'], 
                                 reindex_y_full['numu'],
                                 reindex_y_full['nutau']], ignore_index=True, sort=False)
    
    print("After Reduction  :\n")

    for part_type in ['nuel', 'numu', 'nutau']:
        print("Particle type: " + part_type)
        print("  TT_df : " + str(len(reindex_TT_df [part_type])))
        print("  y_full: " + str(len(reindex_y_full[part_type])))

    print()
    print("Combined TT_df : " + str(len(combined_TT_df)))
    print("Combined y_full: " + str(len(combined_y_full)))
    
    return combined_TT_df, combined_y_full


def normalise_target_energy(reindex_y_full, norm = 1. / 4000):
    # True value of NRJ for each true Nue event
    reindex_y_full[["E"]] *= norm
    return reindex_y_full


def read_pickled_df(detector_params, path_nuel, path_numu, path_nutau):
    reidx_TT_df, reidx_y_full = load_dataframes(detector_params, path_nuel, path_numu, path_nutau)
    
    # not required
    # reidx_TT_df, reidx_y_full = balance_events_num(reidx_TT_df, reidx_y_full)
    merged_TT_df, merged_y_full = merge_events_arrays(reidx_TT_df, reidx_y_full)
    
    merged_y_full = normalise_target_energy(merged_y_full)
    
    return merged_TT_df, merged_y_full