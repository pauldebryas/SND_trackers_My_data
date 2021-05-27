import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pylab as plt
import scipy.stats as stats


def plot_2d_energy_hist(X_arr, y_true, y_pred, model_name):
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
    
    
def plot_2d_energy_hist_clip(X_full, y_true, y_pred, clip, model_name):
    fig, ax = plt.subplots(figsize=(8,6))

    plt.xlim(0,None)
    
    hist = ax.hist2d(X_full[:, 0], y_true[:, 0], 
                     bins=100, norm=mpl.colors.LogNorm(), vmax=150)
    
    plt.xlabel('pixel sum')
    plt.ylabel('normalised energy')

    plt.axvline(x=clip, c='m', alpha=0.9, label='Min clip ' + str(clip))

    ax.plot(X_full, y_pred, 'deeppink', marker='.', linestyle='None', alpha=0.3, label=model_name)

    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.set_label('# of particles')

    plt.legend(loc='lower right')
    plt.show()

    

def comp_resolution(y_true, y_pred):
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)
   
    resolution = np.divide(y_pred_flat - y_true_flat, y_true_flat)

    return resolution
    
    
def plot_res_vs_energy(X_arr, y_true, y_pred, bins=100, vmax=150):
    resolution = comp_resolution(y_true, y_pred)
        
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    
    energy, xlabel = None, None
    
    for i in range(2):        
        if i == 0:
            energy = X_arr[:,0]
            xlabel = r'$E_{reco}$'
        else:
            energy = y_true[:,0]
            xlabel = r'$E_{true}$'
        
        axs[i][0].set_ylabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
        hist = axs[i][0].hist2d(energy, resolution, 
                                bins=bins, norm=mpl.colors.LogNorm(), vmax=vmax,
                                range=(None,None))

        axs[i][1].set_ylabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
        hist = axs[i][1].hist2d(energy, resolution, 
                                bins=bins, norm=mpl.colors.LogNorm(), vmax=vmax, 
                                range=(None,(-2.5,20)))

        axs[i][0].set_xlabel(xlabel)
        axs[i][0].grid()
        axs[i][1].set_xlabel(xlabel)
        axs[i][1].grid()
        
    # todo: colorbar
    
    plt.show()
    
    
def plot_res_hist(y_true, y_pred, hist_range=None, bins=100, density=False):
    resolution = comp_resolution(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8,6))

    n, bins, patches = ax.hist(resolution, bins, hist_range, density=density)
    
    plt.xlabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
    plt.ylabel('# particles')

    plt.show()
    

def plot_res_hist_fit(y_true, y_pred, hist_range=None, bins=100):
    resolution = comp_resolution(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8,6))

    n, bins, patches = ax.hist(resolution, bins, hist_range, 
                               density=True, label='Data PDF')
    
    print('mean   = {:.5e}'.format(resolution.mean()))
    print('std    = {:.5e}'.format(resolution.std()))
    print('median = {:.5e}'.format(np.median(resolution)))
    
    (loc, scale) = stats.norm.fit(resolution)
    label_str = "Normal ({:.3e}, {:.3e})".format(loc, scale)
    x = np.linspace(bins[0] - 1, bins[-1] + 1, 300)
    
    plt.plot(x, stats.norm.pdf(x, loc, scale), label=label_str)
    
    plt.xlabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
    plt.ylabel('# particles')

    plt.legend()
    plt.show()