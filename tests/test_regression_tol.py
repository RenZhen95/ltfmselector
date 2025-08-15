import pytest
from ltfmselector import LTFMSelector

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def get_test_data(_data):
    match _data:
        case "california_housing":
            raw_data = fetch_california_housing()
        case _:
            raise ValueError(f"Couldn't find {_data} dataset!")

    # Get data
    X = raw_data['data']

    # Get target
    y = raw_data['target']

    # Get feature names
    feature_names = raw_data['feature_names']

    # Get description
    dataset_description = raw_data['DESCR']

    # Convert data into pandas DataFrame
    df = pd.DataFrame(
        np.c_[X, y], columns = np.append(feature_names, ['target'])
    )

    # Split the dataset for training and test
    X_df = df.drop(['target'], axis=1)
    y_df = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2, random_state=5
    )

    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # Also remember to reset the index of X_train and X_test
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)

    return X_train, y_train, X_test, y_test

def test_regression():
    X_train, y_train, X_test, y_test = get_test_data("california_housing")

    AgentSelector = LTFMSelector(100, pType='regression')
    # Go for 32000 if we got time

    # Now letting the agent train, this could take some time ...
    doc = AgentSelector.fit(X_train, y_train, agent_neuralnetwork=None, lr=1e-5)
