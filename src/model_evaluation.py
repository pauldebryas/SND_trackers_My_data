from sklearn.metrics import *

# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
def get_scores(y_train, y_pred):
    score_func_arr = [explained_variance_score,
                      max_error,
                      mean_absolute_error,
                      mean_squared_error,
                      mean_squared_log_error, 
                      #mean_absolute_percentage_error, # update scikit to use
                      median_absolute_error,
                      r2_score]

    metric_func = explained_variance_score
    
    scores_dict = dict()

    for metric_func in score_func_arr:
        scores_dict[metric_func.__name__] = metric_func(y_train, y_pred)
        print(metric_func.__name__, metric_func(y_train, y_pred))
        
    return scores_dict