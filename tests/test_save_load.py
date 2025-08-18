import pytest
from ltfmselector import LTFMSelector
from ltfmselector import load_model

from utils_fortesting import get_test_data

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.svm import SVR

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, dataset_shape):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(dataset_shape[1], 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

class SimpleTorchModel():
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def fit(self, X, y, sample_weight=None):
        self.model = SimpleNeuralNetwork(X.shape).to("cpu")
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=5e-5, weight_decay=1.0
        )

        for epoch in range(256):
            y_pred = self.model(
                torch.from_numpy(X.astype(np.float32))
            )
            loss = self.loss_fn(
                y_pred, torch.from_numpy(y.astype(np.float32))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.model.eval()

        with torch.no_grad():
            ypred = self.model(
                torch.tensor(X.astype(np.float32)).to("cpu")
            )

        return np.reshape(ypred.cpu().numpy(), -1)

def test_save():
    X_train, y_train, X_test, y_test = get_test_data("california_housing")

    AgentSelector = LTFMSelector(
        100, batch_size=128, pType='regression',
        pModels=[SimpleTorchModel(), SVR()]
    )
    # Go for 32000 if we got time

    # Now letting the agent train, this could take some time ...
    doc = AgentSelector.fit(
        X_train, y_train,
        checkpoint_interval=2,
        agent_neuralnetwork=None, lr=1e-5
    )

    AgentSelector.save_model("saveload_testfile")

def test_load():
    Selector, pModels = load_model("saveload_testfile.tar.gz")

    # Prediction models MUST be loaded manually.
    models = []
    for _m in pModels:
        if _m[0] != "pytorch":
            models.append(_m[1])
        else:
            # If you had a PyTorch model, initialize it and load the
            # saved state_dict
            torchModel = SimpleTorchModel()
            torchModel.load_state_dict(_m[1])
            torchModel.eval()
            models.append(torchModel)

    Selector.pModels = models

    # Short test
    X_train, y_train, X_test, y_test = get_test_data("california_housing")
    ypred, doc_test = Selector.predict(X_test.loc[0:3, :])

    print("Prediction on four test samples:")
    print(ypred)
