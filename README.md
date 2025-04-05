<h1>
  <picture>
    <img alt="LTFMSelectorLogo" src="icons/icon.svg" width="600px">
  </picture>
</h1>

# LTFMSelector
Locally-Tailored Feature and Model Selector with Deep Q-Learning

## Installation
```
pip install ltfmselector
```

## Basic usage
```python
from ltfmselector import LTFMSelector

# Initialize an agent to learn to selects features and models, specifically tailored to each example
AgentSelector = LTFMSelector(<#episodes>, pType=<'classification', 'regression'>)

# Fit
AgentSelector(<X: pd.DataFrame>, <y: pd.Series>)

# Predict
y_pred, doc = AgentSelector.predict(<X_test: pd.DataFrame>)
```

For more examples check out the [examples](https://github.com/RenZhen95/ltfmselector/tree/master/examples).

## Citing LTFMSelector
This library is implemented based on the work presented in this abstract:

J.C. Liaw, C.Z. Chaing, D. Raab, M. Siebler, H. Hefter, D. Zietz, M. Jäger, A. Kecskeméthy, F. Geu Flores. Interdisciplinary Gait Assessment with Patient-Specific Feature and Model Selection via Reinforcement Learning. 11. IFToMM D-A-CH Konferenz 2025, 20./21. Februar 2025, FH Kärnten, Villach. [HTML](https://doi.org/10.17185/duepublico/82941)
