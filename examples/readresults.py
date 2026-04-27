import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

from ltfmselector import load_model

# Example patient: RB_00103_B_10_A02_U02

if len(sys.argv) < 6:
    print(
        "Possible usage: python3.11 readresults.py <modelPath> " +
        "<testDictPkl> <testDataset> <spID> <repGCPkl>"
    )
    sys.exit(1)
else:
    modelPath = Path(sys.argv[1])
    testpkl = Path(sys.argv[2])
    testDataset = Path(sys.argv[3])
    spID = sys.argv[4]

    repGCPkl = Path(sys.argv[5])

# # Load trained agent
# Selector, pModels = load_model(modelPath, nEpisodes=None)
# dataset = Selector.X

# # First initialize background dataset for SHAP
# # 1. Extracting the representative samples for each SMS-subscore group
# with open(repGCPkl, "rb") as handle:
#     repGCDict = pickle.load(handle)

# # 2. Get 90 samples with a similar subscore distribution during training
# nTotalSamples = 90
# nSubscore0 = round((repGCDict[0].shape[0]/dataset.shape[0])*nTotalSamples)
# nSubscore1 = round((repGCDict[1].shape[0]/dataset.shape[0])*nTotalSamples)
# nSubscore2 = round((repGCDict[2].shape[0]/dataset.shape[0])*nTotalSamples)
# nSubscore3 = round((repGCDict[3].shape[0]/dataset.shape[0])*nTotalSamples)
# nSubscore = np.array((nSubscore0, nSubscore1, nSubscore2, nSubscore3))

# if nSubscore.sum() < nTotalSamples:
#     nSubscore[np.argmin(nSubscore)] = nSubscore[np.argmin(nSubscore)] + (nTotalSamples - nSubscore.sum())
# elif nSubscore.sum() > nTotalSamples:
#     nSubscore[np.argmax(nSubscore)] = nSubscore[np.argmax(nSubscore)] - (nSubscore.sum() - nTotalSamples)

# # Sampling 90 samples as background dataset for masking (averaged out)
# samples0 = repGCDict[0].iloc[0:nSubscore[0]]
# samples1 = repGCDict[1].iloc[0:nSubscore[1]]
# samples2 = repGCDict[2].iloc[0:nSubscore[2]]
# samples3 = repGCDict[3].iloc[0:nSubscore[3]]
# samplesMasking = list(samples0.index) + list(samples1.index) + list(samples2.index) + list(samples3.index)

# datasetMask = dataset.loc[samplesMasking, :]
# datasetMask.to_csv("StabilityBG_repGC_n90.dat", sep=' ')

# sys.exit()

# ================================================
# This should all be implemented into ltfmselector
# ================================================
# Load trained agent
Selector, pModels = load_model(modelPath, nEpisodes=None)
trainDF = Selector.X

# Load test dataset
with open(testpkl, 'rb') as handle:
    testDict = pickle.load(handle)

# Get prediction to explain
spDict = testDict[spID]

# Get PM used
pModel = pModels[spDict["PredModel"]][1]
print(f"Prediction model : {pModel}")

# Get features used
featuresMask = spDict["Mask"]
print(f"Features mask    :\n{featuresMask}")

testDF = pd.read_table(testDataset, sep=' ', index_col=0)

# Get trimmed datasets
trimmedTraintDF = trainDF.iloc[:, np.where(featuresMask==1.)[0].tolist()]
print(trimmedTraintDF)
trimmedTestDF = testDF.iloc[:, np.where(featuresMask==1.)[0].tolist()]
print(trimmedTestDF)

sys.exit(0)
