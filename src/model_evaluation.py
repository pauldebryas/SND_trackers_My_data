from sklearn.metrics import *
import pandas as pd
import numpy as np
import os


SCORES_DIR = 'results/scores/'
TEST_SCORES_DIR = 'results/test_scores/'
#PRED_DIR = 'results/pred/'
#MODELS_DIR = 'results/models/'


def comp_resolution(y_true, y_pred):
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)
   
    resolution = np.divide(y_pred_flat - y_true_flat, y_true_flat)

    return resolution


def get_scores(y_train, y_pred):
    # https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    
    score_func_arr = [explained_variance_score,
                      max_error,
                      mean_absolute_error,
                      mean_squared_error,
                      #mean_squared_log_error, # negative values may appear
                      #mean_absolute_percentage_error, # update scikit to use
                      median_absolute_error,
                      r2_score]

    metric_func = explained_variance_score
    
    scores_dict = dict()

    for metric_func in score_func_arr:
        scores_dict[metric_func.__name__] = metric_func(y_train, y_pred)
        print(metric_func.__name__, metric_func(y_train, y_pred))
        
    resolution_arr = comp_resolution(y_train, y_pred)
    scores_dict['avg_resolution'] = resolution_arr.mean()
    scores_dict['std_resolution'] = resolution_arr.std()

    return scores_dict


def save_scores(y_train, y_pred, save_file_prefix, folder_path=SCORES_DIR):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    scores_dict = get_scores(y_train, y_pred)
    scores_df = pd.DataFrame.from_dict(scores_dict, orient = 'index')
    scores_df = scores_df.rename(columns={0 : save_file_prefix})
    #scores_df = scores_df.T    
    
    file_path = os.path.join(folder_path, save_file_prefix + '_scores.csv')
    scores_df.to_csv(file_path, index=True)
        
    return scores_df
    

def collect_all_scores(folder_path=SCORES_DIR):
    # get list of all files in a directory
    files_arr = os.listdir(folder_path)

    # files and dirs with full path
    files_arr = [os.path.join(folder_path, f) for f in files_arr]

    # select only files
    files_arr = [f for f in files_arr if os.path.isfile(f)]
    
    # collect all df with scores
    all_score_df = pd.DataFrame()
    first_file = True

    for file in files_arr:
        new_scores = pd.read_csv(file)

        if first_file:
            all_score_df['Score'] = new_scores[new_scores.keys()[0]].to_numpy()
            first_file = False

        new_col_name = new_scores.keys()[1]
        all_score_df[new_col_name] = new_scores[new_col_name].to_numpy()

    return all_score_df


'''
def save_predictions(y_true, y_pred, save_file_prefix, folder_path=PRED_DIR):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    pred_dict = {'true' : y_true,
                 'pred' : y_pred}
    
    scores_df = pd.DataFrame.from_dict(scores_dict, orient = 'index')
    
    file_path = os.path.join(folder_path, save_file_prefix + '_scores.csv')
    scores_df.to_csv(file_path, index=True)

    return pred_dict
'''