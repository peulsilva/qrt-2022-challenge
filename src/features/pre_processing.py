import pandas as pd

def reshape_X(X : pd.DataFrame):
    X_reshape = pd.concat([ 
        X.transpose()\
            .shift(i+1)\
            .stack(dropna=False) for i in range(250) ], 
        1)\
        .dropna()
    X_reshape.columns = pd.Index(range(1,251), name='timeLag')

    return X_reshape