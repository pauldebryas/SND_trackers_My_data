import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pylab as plt


def plot_2d_energy_hist(X_arr, y_true, y_pred, model_name='L2 regression'):
    fig, ax = plt.subplots(figsize=(8,6))

    hist = ax.hist2d(X_arr[:, 0], y_true[:, 0], 
                     bins=100, norm=mpl.colors.LogNorm(), vmax=150)
    
    plt.xlabel('pixel sum')
    plt.ylabel('normalised energy')

    #plt.axvline(x=min_clip, c='m', alpha=0.9, label='Min clip ' + str(min_clip))

    ax.plot(X_arr, y_pred, 'deeppink', marker='.', linestyle='None', alpha=0.3, label=model_name)

    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.set_label('# of particles')

    plt.legend(loc='lower right')
    plt.show()
    
    
def plot_res_vs_energy(X_arr, y_true, y_pred):
    resolution = np.divide(y_pred - y_true, y_true)
    
    fig, axs = plt.subplots(2,2, figsize=(12,8))

    for i in range(2):
        energy = None
        xlabel = None
        
        if i == 0:
            energy = X_arr[:, 0]
            xlabel = r'$E_{reco}$'
        else:
            energy = y_true[:, 0]
            xlabel = r'$E_{true}$'
        
        hist = axs[i][0].hist2d(energy, resolution[:, 0], 
                          bins=100, norm=mpl.colors.LogNorm(), vmax=150)

        axs[i][0].set_ylim(-2, None)

        axs[i][0].set_xlabel(xlabel)
        axs[i][0].set_ylabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
        axs[i][0].grid()


        hist = axs[i][1].hist2d(energy, resolution[:, 0], 
                          bins=100, norm=mpl.colors.LogNorm(), vmax=150)

        axs[i][1].set_ylim(-2, 20)

        axs[i][1].set_xlabel(xlabel)
        axs[i][1].set_ylabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
        axs[i][1].grid()
    
    plt.show()
    
    
def plot_res_hist(y_true, y_pred, hist_range=None):
    resolution = np.divide(y_pred - y_true, y_true)
    
    fig, ax = plt.subplots(figsize=(8,6))

    ax.hist(resolution.reshape(-1), bins=100, range=hist_range)
    
    plt.xlabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
    plt.ylabel('# particles')

    plt.show()