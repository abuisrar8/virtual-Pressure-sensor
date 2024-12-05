import numpy as np


def MAX_ABSOLUTE_ERROR(true_val,pred_val):
    return np.max(np.abs(true_val - pred_val))
def MAE(true_val, pred_val):
   
    return np.mean(np.abs(true_val - pred_val))
def L2_ERROR(true_val,pred_val):
    
    return np.linalg.norm(true_val-pred_val,2)/np.linalg.norm(true_val,2)

def MSE(true_val,pred_val):
    
    return  np.mean(np.square(true_val-pred_val))

def RMSE(true_val,pred_val):
    
    return  np.sqrt(np.mean(np.square(true_val-pred_val)))

def R2_SCORE(true_val,pred_val):
    
    mean_true = np.mean(true_val)
    
    return 1.0 - np.mean(np.square(true_val-pred_val))/np.mean(np.square(true_val-mean_true))
