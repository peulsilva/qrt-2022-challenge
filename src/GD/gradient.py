import numpy as np
import pandas as pd
def gradient(A:np.ndarray, 
             beta: np.ndarray,
             X: pd.DataFrame,
             y: pd.DataFrame, 
             f: callable, 
             X_reshape: pd.DataFrame,):
    """Computes the gradient of a function f(A, beta, X, y)

    Args:
        A (np.ndarray): _description_
        beta (np.ndarray): _description_
        X (pd.DataFrame): _description_
        y (pd.DataFrame): _description_
        f (callable): objective function
        X_reshape (pd.DataFrame): _description_

    Returns:
        np.ndarray: Gradient (df/dx1, df/dx2, ..., df/dx10)
    """            

    gradient = []

    for i in range(len(beta)):

        delta_xi = beta[i]*1e-5
        beta_db = beta.copy()
        beta_db[i] += delta_xi 
        df_dxi = (f(A, beta_db, X, y,  X_reshape) - f(A, beta, X, y, X_reshape))/delta_xi

        gradient.append(df_dxi)
    
    gradient = np.array(gradient)
    gradient[np.isnan(gradient)] = 0

    return gradient