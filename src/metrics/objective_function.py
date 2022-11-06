import numpy as np
import pandas as pd
from src.metrics.custom_metric_QRT_22_6BhdSkn import transform_submission_to_ypred, \
    metric

def objective (A : np.ndarray, 
               beta: np.ndarray, 
               X : pd.DataFrame, 
               y: pd.DataFrame, 
               X_reshape : pd.DataFrame = None):
    """Function to be maximized on Gradient descent

    Args:
        A (np.ndarray): Orthonormal matrix A
        beta (np.ndarray): Feature importance  
        X (pd.DataFrame): _description_
        y (pd.DataFrame): _description_
        X_reshape (pd.DataFrame, optional): _description_. Defaults to None.

    Returns:
        float : Metric calculated over (A, beta)
    """    

    y_pred = transform_submission_to_ypred(A, beta, X, y, X_reshape)
    return metric(y, y_pred)