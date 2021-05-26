import root_numpy
import sys
import os
import pickle
import numpy as np
import pandas as pd
import json
from bisect import bisect_left
#from matplotlib import pylab as plt

CM_TO_MUM = 1e4


class Parameters(object):
    """
    Class to store all parameters of the geometry configuration
    """
    def __init__(self, configuration: str):
        """
        :param configuration: String of used config: 10X0, 9X0, 6X0, 5X0
        """
        self.configuration = configuration
        with open("parameters/parameters.json", "r") as f:
            self.snd_params = json.load(f)
        self.scifi_tt_positions = [item for sublist in self.snd_params[configuration]['SciFi_tracker']['TT_POSITIONS']
                for item in sublist]
        self.mu_upstream_tt_positions = [item for sublist in self.snd_params[configuration]['Mu_tracker_upstream']['TT_POSITIONS']
                for item in sublist]
        self.mu_downstream_tt_positions = [item for sublist in self.snd_params[configuration]['Mu_tracker_downstream']['TT_POSITIONS']
                for item in sublist]
        # This is used to map index of binary search by Z to index of TT number
        self.tt_map = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7}



class DataPreprocess(object):
    def __init__(self, parameters):
        self.params = parameters

    def open_shower_file(self, filename, start=0, stop=100, step=1):
        """
        Read root file to numpy. Quite slow for big amount of data.
        :param filename:
        :param start:
        :param stop:
        :param step:
        :return:
        """
        prefixMC = 'MCTrack'
        prefixSciFiTT = 'ScifiPoint'
        prefixMuFilterPoint = 'MuFilterPoint'
        showers_data_root = root_numpy.root2array(filename, treename='cbmsim', start=start, stop=stop, step=step,
                                                  branches=[prefixMC + '.fPx',
                                                            prefixMC + '.fPy',
                                                            prefixMC + '.fPz',
                                                            prefixMC + '.fStartX',
                                                            prefixMC + '.fStartY',
                                                            prefixMC + '.fStartZ',
                                                            prefixMC + '.fMotherId',
                                                            prefixMC + '.fPdgCode',
                                                            prefixSciFiTT + '.fPx',
                                                            prefixSciFiTT + '.fPy',
                                                            prefixSciFiTT + '.fPz',
                                                            prefixSciFiTT + '.fX',
                                                            prefixSciFiTT + '.fY',
                                                            prefixSciFiTT + '.fZ',
                                                            prefixSciFiTT + '.fPdgCode',
                                                            prefixMuFilterPoint + '.fPx',
                                                            prefixMuFilterPoint + '.fPy',
                                                            prefixMuFilterPoint + '.fPz',
                                                            prefixMuFilterPoint + '.fX',
                                                            prefixMuFilterPoint + '.fY',
                                                            prefixMuFilterPoint + '.fZ',
                                                            prefixMuFilterPoint + '.fPdgCode'
                                                            ],
                                                  )
        return showers_data_root

    def extract_showers(self, showers_data_root, E_TRHESHOLD=0.0001):
        """
        Convert root_numpy array to dict of MC true info and responses of the SciFi/Mu TT.
        Remove low energy events and events from upstream TT.
        :param showers_data_root: root_numpy array
        :param E_TRHESHOLD: Energy cutoff
        :return: dict of TT responses, dict of MC true info, indices of events
        """
        MC_info = []
        SciFi_info = []
        Mu_info = []
        initial_indeces = []

        no_ele, low_energy = 0, 0
        for index, shower_data_root in enumerate(showers_data_root):
            # extract data
            fPx_mc, fPy_mc, fPz_mc, fStartX_mc, fStartY_mc, fStartZ_mc, fMotherId_mc, \
            fPdgCode_mc, \
            fPx_sifi, fPy_sifi, fPz_sifi, fStartX_sifi, fStartY_sifi, fStartZ_sifi, \
            fPdgCode_sifi, \
            fPx_mu, fPy_mu, fPz_mu, fStartX_mu, fStartY_mu, fStartZ_mu, \
            fPdgCode_mu = \
                shower_data_root

            #Add x, y midpoint
            x_mid_scifi = (self.params.snd_params[self.params.configuration]['SciFi_tracker']['X_max']+self.params.snd_params[self.params.configuration]['SciFi_tracker']['X_min'])/2
            y_mid_scifi = (self.params.snd_params[self.params.configuration]['SciFi_tracker']['Y_max']+self.params.snd_params[self.params.configuration]['SciFi_tracker']['Y_min'])/2
            x_mid_mu = (self.params.snd_params[self.params.configuration]['Mu_tracker_upstream']['X_max']+self.params.snd_params[self.params.configuration]['Mu_tracker_upstream']['X_min'])/2
            y_mid_mu = (self.params.snd_params[self.params.configuration]['Mu_tracker_upstream']['Y_max']+self.params.snd_params[self.params.configuration]['Mu_tracker_upstream']['Y_min'])/2
            fStartX_sifi -= x_mid_scifi
            fStartY_sifi -= y_mid_scifi
            fStartX_mu -= x_mid_mu
            fStartY_mu -= y_mid_mu

            ## CUTS ON ELECTRON
            #ele_mask = np.logical_and(fMotherId_mc == -1, np.abs(fPdgCode_mc) == 12)
            #if ele_mask.sum() == 0:
            #    no_ele += 1
            #    continue
            #elif ele_mask.sum() > 1:
            #    raise

            #if np.sqrt(fPx_mc ** 2 + fPy_mc ** 2 + fPz_mc ** 2)[ele_mask][0] < 0.5:
            #    low_energy += 1
            #    continue


            # Selection for SciFi
            # just full mask
            mask_sifi = np.full_like(fPx_sifi, fill_value=True, dtype=np.bool)
            # 0-length tracks looks bad
            # mask_sifi = mask_sifi & (fLength_sifi != 0)
            # visability mask: Only tracks with P > E_TRHESHOLD GeV are seen in emulson
            mask_sifi = mask_sifi & (np.sqrt(fPx_sifi ** 2 + fPy_sifi ** 2 + fPz_sifi ** 2) > E_TRHESHOLD)
            # Remove hits from upstream TT plane
            mask_sifi = mask_sifi & self.check_position(fStartZ_sifi)
            # Remove back scattered hits
            mask_sifi = mask_sifi & (fPz_sifi >= 0)

            # Selection for mu 
            # just full mask
            mask_mu = np.full_like(fPx_mu, fill_value=True, dtype=np.bool)
            # visability mask: Only tracks with P > E_TRHESHOLD GeV are seen in emulson
            mask_mu = mask_mu & (np.sqrt(fPx_mu ** 2 + fPy_mu ** 2 + fPz_mu ** 2) > E_TRHESHOLD)
            # Remove back scattered hits
            mask_mu = mask_mu & (fPz_mu >= 0)

            SciFi_resp = {
                'PX': fPx_sifi[mask_sifi],
                'PY': fPy_sifi[mask_sifi],
                'PZ': fPz_sifi[mask_sifi],

                'X': fStartX_sifi[mask_sifi],
                'Y': fStartY_sifi[mask_sifi],
                'Z': fStartZ_sifi[mask_sifi],
                'PdgCode': fPdgCode_sifi[mask_sifi]
            }

            Mu_resp = {
                'PX': fPx_mu[mask_mu],
                'PY': fPy_mu[mask_mu],
                'PZ': fPz_mu[mask_mu],

                'X': fStartX_mu[mask_mu],
                'Y': fStartY_mu[mask_mu],
                'Z': fStartZ_mu[mask_mu],
                'PdgCode': fPdgCode_mu[mask_mu]
            }

            mc_info = {
                'PX': fPx_mc,
                'PY': fPy_mc,
                'PZ': fPz_mc,

                'X': fStartX_mc,
                'Y': fStartY_mc,
                'Z': fStartZ_mc,
                'MotherId': fMotherId_mc,
                'PdgCode': fPdgCode_mc
            }

            SciFi_info.append(SciFi_resp)
            Mu_info.append(Mu_resp)
            MC_info.append(mc_info)
            initial_indeces.append(index)
        #print("Number of event that are not nu_e: " + str(no_ele))
        #print("Number of event that are below the energy treshold: " + str(low_energy))    
        #print("length of SciFi_info: "+ str(len(SciFi_info)))
        #print("length of MC_info: " + str(len(MC_info)))
        return SciFi_info, Mu_info, MC_info, initial_indeces

    def check_position(self, z_pos):
        """
        Mask events in upstream TT.
        :param z_pos:
        :return: boolean mask of selected events
        """
        mask = np.full_like(z_pos, fill_value=False, dtype=np.bool)
        for element in self.params.snd_params[self.params.configuration]['SciFi_tracker']['TT_POSITIONS']:
            mask = np.logical_or(mask, np.logical_and(z_pos > element[0], z_pos < element[1]))
        return mask

    def clean_data_and_save(self, showers_data_root, save_folder, time_threshold=20, n_hits_threshold=0):
        r"""
        Apply cuts to events and save DataFrame to pickle format.
        :param showers_data_root: root_numpy array
        :param save_folder: Directory to store files
        :param time_threshold: max :math:`\\delta t` between hits after which event is discarded
        :param n_hits_threshold: Minimum number of hits in all TT station to save event
        :return:
        """
        SciFi_info, Mu_info, MC_info, initial_indeces= self.extract_showers(showers_data_root)
        MC_df = pd.DataFrame(MC_info)
        TT_df = pd.DataFrame(SciFi_info)
        Mu_df = pd.DataFrame(Mu_info)
        #print(MC_df.shape, TT_df.shape)

        # Remove events, that have hits separated with more then 20ns
        # This is silly cut because of absence of true detector digitisation
        # max_min_time = TT_df['Time'].map(lambda x: np.max(x) - np.min(x) if len(x) != 0 else -1)
        # TT_df = TT_df[max_min_time < time_threshold]
        # MC_df = MC_df[max_min_time < time_threshold]
        # print(MC_df.shape)

        # Remove events with less or equal then threshold hits in TT
        n_hits = TT_df.X.map(lambda x: len(x))
        TT_df = TT_df[n_hits >= n_hits_threshold]
        MC_df = MC_df[n_hits >= n_hits_threshold]
        Mu_df = Mu_df[n_hits >= n_hits_threshold]
        #print(MC_df.shape)

        # Select MC showers, which have passed the cuts
        indeces = MC_df.index.values

        nu_params = [[],[]]
        for counter, index in enumerate(indeces):
            ele_mask = np.logical_and(MC_info[index]["MotherId"] == -1, np.abs(MC_info[index]["PdgCode"]) != 0)
            nu_params[0].append(np.linalg.norm([MC_info[index][P][ele_mask] for P in ['PX', 'PY', 'PZ']]))
            nu_params[1].append(int(-1))
        nu_params = pd.DataFrame(np.array(nu_params).T, columns=["E","Label"])
        TT_df.to_pickle(os.path.join(save_folder, "tt_cleared.pkl"))
        Mu_df.to_pickle(os.path.join(save_folder, "mu_cleared.pkl"))
        nu_params.to_pickle(os.path.join(save_folder, "y_cleared.pkl"))

