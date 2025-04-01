import torch
import pickle
import os, sys
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV

# from stdutils.MLPRegressor import MLPRegressor

from LTFMSelector import LTFMSelector

# Set manual seed
torch.manual_seed(12)

# Function to get patient's ID given a stride pair ID
getPatientID = lambda x: x[0:5] if x.startswith("ES") else x[0:8]

if len(sys.argv) < 4:
    print("Possible usage: python test.py <X> <y> <subscore> <X_test>")
    sys.exit(1)
else:
    train_dataset = sys.argv[1]
    medscores = sys.argv[2]
    subscore = sys.argv[3]
    test_dataset = sys.argv[4]

X = pd.read_table(train_dataset, sep=' ', index_col='StridePairID')
print(X)

X_test = pd.read_table(test_dataset, sep=' ', index_col='StridePairID')
print(X_test)

medscores = pd.read_excel(medscores, index_col='ID')

y = pd.Series(data=np.zeros(X.shape[0]), index=X.index)
for i in y.index:
    y.at[i] = medscores.at[getPatientID(i), subscore]

y_test = pd.Series(data=np.zeros(X_test.shape[0]), index=X_test.index)
for i in y_test.index:
    y_test.at[i] = medscores.at[getPatientID(i), subscore]

# learning_models = [
#     GridSearchCV(
#         RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1),
#         {"max_depth": [3, 4, 5]},
#         n_jobs=-1, cv=KFold(n_splits=3, shuffle=False),
#         scoring='r2', verbose=3
#     ),
#     GridSearchCV(
#         SVR(kernel="rbf"),
#         {"C": [0.1, 0.5, 1.0]},
#         n_jobs=-1, cv=KFold(n_splits=3, shuffle=False),
#         scoring='r2', verbose=3
#     ),
#     GridSearchCV(
#         MLPRegressor(epochs=300, batch_size=128, learning_rate=1e-4, device="cpu"),
#         {"weight_decay": [0.01, 0.1], 'dropout_prob': [0.01, 0.05]},
#         n_jobs=-1, cv=KFold(n_splits=3, shuffle=False),
#         scoring='r2', verbose=3
#     )
# ]

import time
start_time = time.time()

SmartAgentSelector = LTFMSelector(5, batch_size=256, pModels=None)
SmartAgentSelector.fit(X, y, smsproject=True)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds for 1500 episodes")

