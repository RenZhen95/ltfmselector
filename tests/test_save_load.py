import pytest
from ltfmselector import LTFMSelector

from utils_fortesting import get_test_data

def test_regression():
    X_train, y_train, X_test, y_test = get_test_data("california_housing")

    AgentSelector = LTFMSelector(10, pType='regression')
    # Go for 32000 if we got time

    # Now letting the agent train, this could take some time ...
    doc = AgentSelector.fit(X_train, y_train, agent_neuralnetwork=None, lr=1e-5)

    print(AgentSelector.__getstate__())
    print(AgentSelector.env.__getstate__())
