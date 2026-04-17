import os, sys
import numpy as np
import pandas as pd
from utils import get_probInitialFeature, sample_initialFeature

X = pd.read_csv(sys.argv[1], sep=' ', index_col=0)
y = pd.Series(np.zeros(X.shape[0]), index=X.index)

getPatientID = lambda x: x[0:5] if x.startswith('ES') else x[0:8]

medscores = pd.read_excel(sys.argv[2], index_col="ID")
for i in y.index:
    y.at[i] = medscores.at[getPatientID(i), sys.argv[3]]

featureProbs = get_probInitialFeature(
    X, y, n_estimators=10000, random_state=12, n_jobs=-1, pType="regression"
)

for i in range(20):
    print(sample_initialFeature(featureProbs))
