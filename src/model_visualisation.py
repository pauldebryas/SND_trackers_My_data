import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pylab as plt
import scipy.stats as stats


IMG_DIR    = 'results/img/'
IMG_SHOW   = True
IMG_FORMAT = 'png'
IMG_DPI    = 300


# todo: move to a more sensible module
def comp_resolution(y_true, y_pred):
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)
   
    resolution = np.divide(y_pred_flat - y_true_flat, y_true_flat)

    return resolution


# decorator for saving the figures from plotting functions 
def savefig_deco(plot_func, folder_path=IMG_DIR, dpi=IMG_DPI, format=IMG_FORMAT, show=IMG_SHOW):
    
    def wrapper(*args, **kwargs):
        # get the file postfix (if it exists) from the arguments
        if 'save_file_prefix' in kwargs.keys():
            save_file_prefix =  kwargs['save_file_prefix']
            kwargs.pop('save_file_prefix', None)

        # check if folder exists and create if it does not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # form the filename
        save_path = os.path.join(folder_path, save_file_prefix)
        save_path = save_path + '_' + plot_func.__name__
        save_path = save_path + '.' + IMG_FORMAT

        # plot + save + show the image
        fig = plot_func(*args, **kwargs)
        plt.savefig(save_path, dpi=dpi, format=format)
        if show:
            plt.show()
            
        return #fig
   
    return wrapper
    

@savefig_deco
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
                
    return fig
    
    
@savefig_deco
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
    return fig
        
    
@savefig_deco
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
    
    return fig
    
    
@savefig_deco
def plot_res_hist(y_true, y_pred, hist_range=None, bins=100, density=False):
    resolution = comp_resolution(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8,6))

    n, bins, patches = ax.hist(resolution, bins, hist_range, density=density)
    
    plt.xlabel(r'$(E_{reco} - E_{true})~/~E_{true}$')
    plt.ylabel('# particles')

    return fig
    

@savefig_deco
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
    return fig