import root_numpy
import sys
import os
import pickle
import numpy as np
import pandas as pd
import json
from bisect import bisect_left

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
        with open("parameters.json", "r") as f:
            self.snd_params = json.load(f)
        self.tt_positions_ravel = [item for sublist in self.snd_params[configuration]['TT_POSITIONS']
                                   for item in sublist]
        # This is used to map index of binary search by Z to index of TT number
        self.tt_map = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6}


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
        prefixTargetPoint = 'TTPoint'
        showers_data_root = root_numpy.root2array(filename, treename='cbmsim', start=start, stop=stop, step=step,
                                                  branches=[prefixMC + '.fPx',
                                                            prefixMC + '.fPy',
                                                            prefixMC + '.fPz',
                                                            prefixMC + '.fStartX',
                                                            prefixMC + '.fStartY',
                                                            prefixMC + '.fStartZ',
                                                            prefixMC + '.fMotherId',
                                                            prefixMC + '.fM',
                                                            prefixMC + '.fStartT',
                                                            prefixMC + '.fPdgCode',
                                                            prefixTargetPoint + '.fPx',
                                                            prefixTargetPoint + '.fPy',
                                                            prefixTargetPoint + '.fPz',
                                                            prefixTargetPoint + '.fX',
                                                            prefixTargetPoint + '.fY',
                                                            prefixTargetPoint + '.fZ',
                                                            prefixTargetPoint + '.fTime',
                                                            prefixTargetPoint + '.fLength',
                                                            prefixTargetPoint + '.fELoss',
                                                            prefixTargetPoint + '.fDetectorID',
                                                            prefixTargetPoint + '.fTrackID',
                                                            prefixTargetPoint + '.fPdgCode'],
                                                  )
        return showers_data_root

    def extract_showers(self, showers_data_root, E_TRHESHOLD=0.0001):
        """
        Convert root_numpy array to dict of MC true info and responses of the TT.
        Remove low energy events and events from upstream TT.
        :param showers_data_root: root_numpy array
        :param E_TRHESHOLD: Energy cutoff
        :return: dict of TT responses, dict of MC true info, indices of events
        """
        showers_mc = []
        TT_sim = []
        initial_indeces = []

        no_ele, out_of_tt, low_energy = 0, 0, 0
        for index, shower_data_root in enumerate(showers_data_root):
            # extract data
            fPx_mc, fPy_mc, fPz_mc, fStartX_mc, fStartY_mc, fStartZ_mc, fMotherId_mc, \
            fM_mc, fStartT_mc, fPdgCode_mc, \
            fPx_sim, fPy_sim, fPz_sim, fStartX_sim, fStartY_sim, fStartZ_sim, fTime_sim, fLength_sim, \
            fELoss_sim, fDetectorID_sim, fTrackID_sim, fPdgCode_sim = \
                shower_data_root

            ## CUTS ON ELECTRON
            ele_mask = np.logical_and(fMotherId_mc == -1, np.abs(fPdgCode_mc) == 11)
            if ele_mask.sum() == 0:
                no_ele += 1
                continue
            elif ele_mask.sum() > 1:
                raise

            if np.sqrt(fPx_mc ** 2 + fPy_mc ** 2 + fPz_mc ** 2)[ele_mask][0] < 0.5:
                low_energy += 1
                continue

            # just full mask
            mask_sim = np.full_like(fPx_sim, fill_value=True, dtype=np.bool)

            # 0-length tracks looks bad
            # mask_sim = mask_sim & (fLength_sim != 0)

            # visability mask: Only tracks with P > E_TRHESHOLD GeV are seen in emulson
            mask_sim = mask_sim & (np.sqrt(fPx_sim ** 2 + fPy_sim ** 2 + fPz_sim ** 2) > E_TRHESHOLD)

            # Remove hits from upstream TT plane
            mask_sim = mask_sim & self.check_position(fStartZ_sim)

            # Remove back scattered hits
            mask_sim = mask_sim & (fPz_sim >= 0)

            TT_resp = {
                'PX': fPx_sim[mask_sim],
                'PY': fPy_sim[mask_sim],
                'PZ': fPz_sim[mask_sim],

                'X': fStartX_sim[mask_sim],
                'Y': fStartY_sim[mask_sim],
                'Z': fStartZ_sim[mask_sim],
                'Time': fTime_sim[mask_sim],
                'PdgCode': fPdgCode_sim[mask_sim],
                'AssociatedMCParticle': fTrackID_sim[mask_sim],
                'ELoss': fELoss_sim[mask_sim]
            }

            shower_mc = {
                'PX': fPx_mc,
                'PY': fPy_mc,
                'PZ': fPz_mc,

                'X': fStartX_mc,
                'Y': fStartY_mc,
                'Z': fStartZ_mc + np.random.uniform(1e-5, 2 * 1e-5),
                'MotherId': fMotherId_mc,
                'PdgCode': fPdgCode_mc
            }

            TT_sim.append(TT_resp)
            showers_mc.append(shower_mc)
            initial_indeces.append(index)
        print(no_ele, out_of_tt, low_energy)
        return TT_sim, showers_mc, initial_indeces

    def check_position(self, z_pos):
        """
        Mask events in upstream TT.
        :param z_pos:
        :return: boolean mask of selected events
        """
        mask = np.full_like(z_pos, fill_value=False, dtype=np.bool)
        for element in self.params.snd_params[self.params.configuration]['TT_POSITIONS']:
            mask = np.logical_or(mask, np.logical_and(z_pos > element[0], z_pos < element[1]))
        return mask

    def clean_data_and_save(self, showers_data_root, save_folder, time_threshold=20, n_hits_threshold=5):
        r"""
        Apply cuts to evnets and save DataFrame to pickle format.
        :param showers_data_root: root_numpy array
        :param save_folder: Directory to store files
        :param time_threshold: max :math:`\\delta t` between hits after which event is discarded
        :param n_hits_threshold: Minimum number of hits in all TT station to save event
        :return:
        """
        showers_sim, showers_mc, initial_indeces= self.extract_showers(showers_data_root)
        MC_df = pd.DataFrame(showers_mc)
        TT_df = pd.DataFrame(showers_sim)
        print(MC_df.shape, TT_df.shape)

        # Remove events, that have hits separated with more then 20ns
        # This is silly cut because of absence of true detector digitisation
        # max_min_time = TT_df['Time'].map(lambda x: np.max(x) - np.min(x) if len(x) != 0 else -1)
        # TT_df = TT_df[max_min_time < time_threshold]
        # MC_df = MC_df[max_min_time < time_threshold]
        # print(MC_df.shape)

        # Remove events with less or equal then threshold hits in TT
        n_hits = TT_df.X.map(lambda x: len(x))
        TT_df = TT_df[n_hits > n_hits_threshold]
        MC_df = MC_df[n_hits > n_hits_threshold]
        print(MC_df.shape)

        # Select MC showers, which have passed the cuts
        indeces = MC_df.index.values

        nu_params = [[], [], [], []]
        for counter, index in enumerate(indeces):
            ele_mask = np.logical_and(showers_mc[index]["MotherId"] == -1, np.abs(showers_mc[index]["PdgCode"]) == 11)
            nu_params[0].append(np.linalg.norm([showers_mc[index][P][ele_mask] for P in ['PX', 'PY', 'PZ']]))
            nu_params[1].append(showers_mc[index]['Z'][ele_mask][0] -
                                self.params.snd_params[self.params.configuration]["END_OF_BRICK"])
            nu_params[2].append(showers_mc[index]['X'][ele_mask][0])
            nu_params[3].append(showers_mc[index]['Y'][ele_mask][0])
        nu_params = pd.DataFrame(np.array(nu_params).T, columns=["E", "Z", "X", "Y"])

        TT_df.to_pickle(os.path.join(save_folder, "tt_cleared.pkl"))
        nu_params.to_pickle(os.path.join(save_folder, "y_cleared.pkl"))
