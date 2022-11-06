import pandas as pd
import numpy as np

def read_train_test_df():
    X = pd.read_csv('data/X_train_YG7NZSq.csv', index_col=0)
    X.columns.name = 'date'

    y = pd.read_csv('data/Y_train_wz11VM6.csv', index_col=0)
    y.columns.name = 'date'

    return X, y 

def reshape_X(X : pd.DataFrame):
    """Function to reshape the train dataframe

    Args:
        X (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """    
    X_reshape = pd.concat([ 
        X.transpose()\
            .shift(i+1)\
            .stack(dropna=False) for i in range(250) ], 
        1)\
        .dropna()
    X_reshape.columns = pd.Index(range(1,251), name='timeLag')

    return X_reshape

def train_test_split(X: pd.DataFrame,
                     y: pd.DataFrame,
                     random_state : int = 42,
                     train_proportion : float = 0.7):
    """Divides the dataframe into train and test.
    It divides train_proportion companies to train and 
    1-train_proportion companies to test.

    Args:
        X (pd.DataFrame): Train dataframe
        y (pd.DataFrame): Test dataframe
        random_state (int, optional): _description_. Defaults to 42.
        train_proportion (float, optional): _description_. Defaults to 0.7.

    Returns:
        _type_: X_train, X_test, y_train, y_test
    """                     

    train_size = int(train_proportion*len(X))
    random_generator = np.random.RandomState(random_state)
    indexes = random_generator.choice(len(X), 
                                      train_size, 
                                      replace=False
                                    )

    X_train, X_test = X.loc[indexes], X.loc[X.index.difference(indexes)] 
    y_train, y_test = y.loc[indexes], y.loc[y.index.difference(indexes)]

    return X_train, X_test, y_train, y_test