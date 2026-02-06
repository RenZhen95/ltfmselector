import pickle
import random
random.seed(7)
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

from ltfmselector import LTFMSelector

import torch

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV

from predictionModels import MLPRegressor

# Set manual seed
torch.manual_seed(12)

# Function to get patient's ID given a stride pair ID
getPatientID = lambda x: x[0:5] if x.startswith("ES") else x[0:8]

# === === === ===
# Parameters
# === === === ===
# BATCH_SIZE = {128, 256, 512}
# - Number of transitions sampled from the replay buffer to train the policy
# network
BATCH_SIZE = 256

# LR = {1e-3, 1e-4, 1e-5}
# - Learning rate of the AdamW Optimizer
LR = 1e-5

# TAU = {0.1 (lee2021), 0.01, 0.001, 0.0005}
# - Update rate of the TARGET network
TAU = 0.0005

# EPS_START = {0.5 (lee2021), 0.9}
# - Start value of epsilon
EPS_START = 0.9

# EPS_END = {0.00, 0.05 (lee2021)}
# - Final value of epsilon
EPS_END = 0.05

# EPS_DECAY = {1000, 3000}
# - Rate of exponential decay of epsilon (higher means a slower decay)
EPS_DECAY = 1000

# GAMMA is the discount factor as mentioned in the previous section
GAMMA = 0.99

# Total number of episodes the agent experiences
nEpisodes = 5

if len(sys.argv) < 8:
    print(
        "Possible usage: python3 train_wLinearIncrease.py <dataset> <medscores> " +
        "<subscore> <nLayer1> <nLayer2> <saveFileSuffix> <queryCost>"
    )
    sys.exit(1)
else:
    dataset = Path(sys.argv[1])
    medscores = Path(sys.argv[2])
    subscore = sys.argv[3]
    nLayer1 = int(sys.argv[4]) # 1000 (Chiang)
    nLayer2 = int(sys.argv[5]) # 1000 (Chiang)
    saveFileSuffix = sys.argv[6]
    query_cost = float(sys.argv[7])

# If GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = pd.read_table(dataset, sep=' ', index_col=0)
y = pd.Series(index=X.index)

medicalscores = pd.read_excel(medscores, engine="openpyxl", index_col="ID")
for pat in y.index:
    y.at[pat] = medicalscores.at[getPatientID(pat), subscore]

# Initializing the list of regression models
RegressionModels = [
    GridSearchCV(
        SVR(kernel='rbf'),
        {'C': [0.1, 0.5, 1.0]},
        n_jobs=-1, cv=KFold(n_splits=3, shuffle=False),
        scoring='r2', verbose=3
    ),
    GridSearchCV(
        RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1),
        {"max_depth": [3, 4, 5]},
        n_jobs=-1, cv=KFold(n_splits=3, shuffle=False),
        scoring='r2', verbose=3
    ),
    GridSearchCV(
        MLPRegressor(epochs=300, batch_size=128, learning_rate=1e-4, device=device),
        {'weight_decay': [0.01, 0.1], 'dropout_prob': [0.01, 0.05]},
        n_jobs=-1, cv=KFold(n_splits=3, shuffle=False),
        scoring='r2', verbose=3
    )
]

# Initializing the an LTFMSelector object
AgentSelector = LTFMSelector(
    nEpisodes,
    batch_size=BATCH_SIZE,
    tau=TAU,
    eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
    fQueryCost=query_cost, fQueryFunction="linear",
    fThreshold=7, fCap=3.0, fRate=query_cost,
    mQueryCost=query_cost*2,
    fRepeatQueryCost=1.0, p_wNoFCost=5.0, errorCost=1.0,
    pType="regression",
    regression_tol=0.5,
    regression_error_rounding=1,
    pModels=RegressionModels,
    gamma=GAMMA,
    max_timesteps=1000,
    checkpoint_interval=None,
    device=device
)

# Fit
doc, ActionValuesQ = AgentSelector.fit(
    X, y, agent_neuralnetwork=(nLayer1, nLayer2),
    lr=LR, monitor=False, returnQ=True, smsproject=True, log=True
)

# Initialize save file names
saveFileName = f"{subscore}{saveFileSuffix}_bs{BATCH_SIZE}_"
saveFileName += "g{0}_".format(str(GAMMA).replace('.','-'))
saveFileName += "epsS{0}_".format(str(EPS_START).replace('.', '-'))
saveFileName += "epsE{0}_".format(str(EPS_END).replace('.', '-'))
saveFileName += f"epsD{EPS_DECAY}_"
saveFileName += "t{0}_".format(str(TAU).replace('.','-'))
saveFileName += "lr{0}_".format(str(LR).replace('.','-'))
saveFileName += "qC{0}_".format(str(query_cost).replace('.','-'))
saveFileName += f"hL{nLayer1}-{nLayer2}_pLTFM"

# Add list of features of the train dataset
doc["Features"] = X.columns.values

# Add agent parameters
doc["AgentParameters"] = {
    "BATCH_SIZE": BATCH_SIZE,
    "GAMMA": GAMMA,
    "EPS_START": EPS_START, "EPS_END": EPS_END,
    "EPS_DECAY": EPS_DECAY, "TAU": TAU, "LR": LR
}

# Saving the intermediate results during training
savePklDictFileName = f"{saveFileName}_nE{nEpisodes}.pkl_dict"
with open(savePklDictFileName, "wb") as handle:
    pickle.dump(doc, handle)

saveNumpyMatrixFile = f"{saveFileName}_nE{nEpisodes}_ActionValuesQ.npy"
np.save(saveNumpyMatrixFile, ActionValuesQ)

# Saving the LTFMSelector agent object
AgentSelector.save_model(f"{saveFileName}_nE{nEpisodes}")

sys.exit(0)
