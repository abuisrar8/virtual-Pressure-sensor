import numpy as np

def delay_matrix(x,w_sz):
    x_conct = np.array([]).reshape(len(x)-w_sz,0)
    
    for i in range(w_sz,-1,-1):
        x_shift = x[w_sz-i:len(x)-i]
        x_conct = np.concatenate((x_conct,x_shift),axis=1)

    return x_conct

def delay_matrix_features(X,w_sz):
    
    for i in range(X.shape[1]):
        f_delay = delay_matrix(X[:,i:i+1],w_sz)
        if i==0:
            f_delay_mat = f_delay
        else:
            f_delay_mat = np.hstack((f_delay_mat,f_delay))
    return f_delay_mat
    
    
def concatenate_inputs(f,x):
    
    return np.concatenate((f,x),axis=1)

def concatenate_data(*data):
    
    concat = np.array([]).reshape(data[0].shape[0],0)
    for i in range(len(data)):
        concat =  np.hstack((concat,data[i]))
        
    return concat
    
