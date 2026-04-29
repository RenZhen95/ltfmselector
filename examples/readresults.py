import shap
import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import rc
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.size'] = 10

from ltfmselector import load_model

from sklearn.model_selection._search import BaseSearchCV

# From SMS Paper
# sample_forpaper = "ES086-B-3Gang01_A01_U01"  # light
# sample_forpaper = "ES180-BG-3Gang05_A01_U01" # moderate
# sample_forpaper = "RB_00055_B_03_A01_U01"    # moderate
# sample_forpaper = "RB_00100_BG_09_A02_U02"   # significant

if len(sys.argv) < 6:
    print(
        "Possible usage: python3.11 readresults.py <modelPath> " +
        "<testDictPkl> <testDataset> <spID> <bgDataset>"
    )
    sys.exit(1)
else:
    modelPath = Path(sys.argv[1])
    testpkl = Path(sys.argv[2])
    testDataset = Path(sys.argv[3])
    spID = sys.argv[4]
    bgDataset = Path(sys.argv[5])

Selector, pModels = load_model(modelPath, nEpisodes=None)
models = []
for _m in pModels:
    models.append(_m[1])
Selector.pModels = pModels

bgDataset = pd.read_table(bgDataset, sep=' ', index_col=0)
testDF = pd.read_table(testDataset, sep=' ', index_col=0)

_ = Selector.explain_wSHAP(spID, testpkl, testDF, bgDataset, plot=True)
plt.show()

sys.exit(0)
