import numpy as np
import pandas as pd
from src.metrics.check_orthonormality import check_orthonormality
from src.features.pre_processing import reshape_X

def random_A(D=250, F=10):  
    
    M = np.random.randn(D,F)
    randomStiefel = np.linalg.qr(M)[0] # Apply Gram-Schmidt algorithm to the columns of M
    
    return randomStiefel

def fit_beta(A , 
            X_train : pd.DataFrame, 
            y_train : pd.DataFrame):

    X_train_reshape = reshape_X(X_train)

    X_train_reshape.columns = pd.Index(range(1,251), name='timeLag')
    
    features_df = X_train_reshape @ A # the dataframe of the 10 factors created from A with the (date, stock) in index
    target_df = y_train.T.stack()
    beta = np.linalg.inv(features_df.T @ features_df) @ features_df.T @ target_df
    
    return beta.to_numpy()

def metric_train(A, 
                 beta,
                 X: pd.DataFrame,
                 y: pd.DataFrame): 
    
    if not check_orthonormality(A):
        return -1.0    
    X_train_reshape = reshape_X(X)

    Ypred = (X_train_reshape @ A @ beta).unstack().T         
    Ytrue = y
    
    Ytrue = Ytrue.div(np.sqrt((Ytrue**2).sum()), 1)    
    Ypred = Ypred.div(np.sqrt((Ypred**2).sum()), 1)

    meanOverlap = (Ytrue * Ypred).sum().mean()

    return  meanOverlap  

def get_benchmark(X : pd.DataFrame, 
                  y : pd.DataFrame):
    Niter = 1000
    max_metric = -1

    np.random.seed(1234)

    for iteration in range(Niter):
        
        # Generate a uniform random Stiefel matric A and fit beta with minimal mean square prediction error on the training data set
        
        A = random_A()
        beta = fit_beta(A, X, y)
        
        # compute the metric on the training set and keep the best result   
        
        m = metric_train(A, beta, X, y)
            
        if m > max_metric:
            print(iteration, 'metric_train:', m)
            
            max_metric = m
            A_QRT = A
            beta_QRT = beta  

    return max_metric