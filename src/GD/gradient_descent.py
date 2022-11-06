import numpy as np
import pandas as pd 

from src.GD.gradient import gradient

def gradient_descent(A : np.ndarray, 
                     beta: np.ndarray, 
                     X: pd.DataFrame, 
                     y: pd.DataFrame, 
                     f: callable, 
                     X_reshape: pd.DataFrame, 
                     alpha : float = 1.):
    """Performs gradient descent on desired function

    Args:
        A (np.ndarray): _description_
        beta (np.ndarray): _description_
        X (pd.DataFrame): _description_
        y (pd.DataFrame): _description_
        f (callable): objective function
        X_reshape (pd.DataFrame): _description_
        alpha (float, optional): _description_. Defaults to 1..

    Returns:
        _type_: _description_
    """    
    grad = gradient(A, beta, X, y, f, X_reshape)

    return beta + alpha*grad