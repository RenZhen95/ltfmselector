import random
import pandas as pd
from operator import attrgetter
from collections import deque, namedtuple, defaultdict

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Transition object
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory(object):
    '''
    Container of transitions experienced by the agent, with which the policy
    network will be trained with.
    '''
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''
        Save a transition comprised of the:
        1. State, s
        2. Action, a
        3. Next state, s'
        4. Reward (immediate), r
        '''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        '''
        Sample a random batch of training data to train the policy network
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    '''
    Classical Multilayer-Perceptron as the default architecture of the policy
    and target network.
    - 2 hidden layers
    - ReLu activation functions at each layer
    '''
    def __init__(self, n_observations, n_actions, n1=1024, n2=1024):
        super(DQN, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.layer1 = nn.Linear(n_observations, self.n1)
        self.layer2 = nn.Linear(self.n1, self.n2)
        self.layer3 = nn.Linear(self.n2, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return (self.layer3(x))

class scoreGroup:
    '''
    Data container grouping the stride pair IDs according to their designated medical scores
    '''
    def __init__(self, _score, _listOfStridePairs):
        self.Score = _score
        self.StridePairs = _listOfStridePairs
        self.nSamples = len(self.StridePairs)
        self.sampleWeight = 1

    def __str__(self):
        output  = f"Score        : {self.Score}\n"
        output += f"nSamples     : {self.nSamples}\n"
        output += f"sampleWeight : {self.sampleWeight}"

        return output

def balance_classDistribution_patient(template_df, score):
    '''
    Weighing the samples according to
    1. Class distribution
     - Samples are first assigned a score inversely proportional to the class frequency
     - Class count of majority class / Class count of class i
    2. Patient
     - Each stride pair sample is divided by the total number of stride pairs associated
       with that patient

    Parameters
    ----------
    template_df : pd.DataFrame
        DataFrame extracted to be used as a template, mainly for the sample indices and their
        corresponding scores

    score : str
        Medical score to target

    Returns
    -------
    df_sampleWeights : pd.DataFrame
        DataFrame mapping each sample to its corresponding weight
    '''
    # Pandas DataFrame of the weights of each sample
    df_sampleWeights = pd.DataFrame(index=template_df.index, columns=['Weight'])

    # --- 1 --- | Class distribution
    # Dictionary mapping medical score to list of stride pair IDs associated with it
    medscore_stridePairIDs_dict = defaultdict(list)

    for idx in template_df.index:
        medscore_stridePairIDs_dict[template_df.at[idx, score]].append(idx)

    # Initializing the scoreGroup objects
    scoreGroups = []
    for score, vList in medscore_stridePairIDs_dict.items():
        scoreGroups.append(scoreGroup(score, vList))

    # Sorting the scoreGroup objects in ascending order of the number of samples
    scoreGroups.sort(key=lambda x: x.nSamples)

    # Identifying the majority class
    majorityClass = max(scoreGroups, key=attrgetter('nSamples'))

    # The samples in the majority class each gets a weight of 1 (default) and that from the other classes
    # will be assigned weights correspondingly
    for g in scoreGroups:
        g.sampleWeight = majorityClass.nSamples / g.nSamples

        # Assigning the corresponding weight to each stride paid ID in df_sampleWeights
        for stridePair in g.StridePairs:
            df_sampleWeights.at[stridePair, 'Weight'] = g.sampleWeight

    # --- 2 --- | Patient
    # Dictionary mapping patient to number of strides pertaining to that patient
    patient_nStrides_dict = defaultdict(int)

    for idx in template_df.index:
        # First dealing with RehabX patients (ES)
        if idx[0:2] == 'ES':
            patient = ('_').join(idx.split('_')[:-2])
            patient = ('-').join(idx.split('-')[:-1])

        elif idx[0:2] == 'RB':
            patient = ('_').join(idx.split('_')[:-4])

        patient_nStrides_dict[patient] += 1

    for idx in df_sampleWeights.index:
        if idx[0:2] == 'ES':
            patient = ('_').join(idx.split('_')[:-2])
            patient = ('-').join(idx.split('-')[:-1])
            df_sampleWeights.at[idx, 'Weight'] = df_sampleWeights.at[idx, 'Weight'] / patient_nStrides_dict[patient]

        elif idx[0:2] == 'RB':
            patient = ('_').join(idx.split('_')[:-4])
            df_sampleWeights.at[idx, 'Weight'] = df_sampleWeights.at[idx, 'Weight'] / patient_nStrides_dict[patient]

    return df_sampleWeights

def rename_features(_rawFeatureName):
    newfeature = []
    fSplit = _rawFeatureName.split('_')

    if len(fSplit) == 1:
        if "stepTime" in fSplit[0]:
            newfeature.insert(0, "Step Time")
        elif "strideTime" in fSplit[0]:
            newfeature.insert(0, "Stride Time")
        elif "StepFactor" in fSplit[0]:
            newfeature.insert(0, "Step Length (norm.)")
        elif "StrideFactor" in fSplit[0]:
            newfeature.insert(0, "Stride Length (norm.)")
        elif "StepWidth" in fSplit[0]:
            newfeature.insert(0, "Step Width (norm.)")
        elif "sSupportPortion" in fSplit[0]:
            newfeature.insert(0, "Single Support (por.)")
        elif "dSupportPortion" in fSplit[0]:
            newfeature.insert(0, "Double Support (por.)")
        elif "singleSupportTime" in fSplit[0]:
            newfeature.insert(0, "Single Support (time)")
        elif "doubleSupportTime" in fSplit[0]:
            newfeature.insert(0, "Double Support (time)")
        elif "gaitSpeed" in fSplit[0]:
            newfeature.insert(0, "Gait Speed")
        elif "limpIdx" in fSplit[0]:
            newfeature.insert(0, "Limp Index")
        elif "cadence" in fSplit[0]:
            newfeature.insert(0, "Cadence")

        # For AFO and Cane, remove the threshold values
        elif "AFO" in _rawFeatureName:
            newfeature.insert(0, "Ankle-Foot Orthosis?")
        elif "Cane" in _rawFeatureName:
            newfeature.insert(0, "Walking Cane?")

        if fSplit[0][-5:] == "UnAff":
            newfeature.insert(1, "contra.")
        elif fSplit[0][-3:] == "Aff":
            newfeature.insert(1, "ipsi.")

    elif len(fSplit) == 2:
        if "Pelvis" in fSplit[0]:
            newfeature.insert(0, "Pelvis")
        elif "Spine" in fSplit[0]:
            newfeature.insert(0, "Spine")
        elif "Hip" in fSplit[0]:
            newfeature.insert(0, "Hip")
        elif "Knee" in fSplit[0]:
            newfeature.insert(0, "Knee")
        elif "Ankle" in fSplit[0]:
            newfeature.insert(0, "Ankle")
        elif "FootProgression" in fSplit[0]:
            newfeature.insert(0, "Foot Progression")
        elif "Thorax" in fSplit[0]:
            newfeature.insert(0, "Thorax")
        elif "Shoulder" in fSplit[0]:
            newfeature.insert(0, "Shoulder")
        elif "Elbow" in fSplit[0]:
            newfeature.insert(0, "Elbow")
        elif "Wrist" in fSplit[0]:
            newfeature.insert(0, "Wrist")

        if fSplit[0][:3] == "UnA":
            newfeature.insert(2, "contra.")
        elif fSplit[0][:1] == "A":
            newfeature.insert(2, "ipsi.")

        # Axis
        if not "FootProgression" in fSplit[0]:
            if fSplit[0][:3] == "UnA":
                ipsiSide = False
            elif fSplit[0][:1] == "A":
                ipsiSide = True

            if ipsiSide:  # e.g. AxPelvisQD
                relevantIndex = 1
            else:         # e.g. UnAxPelvisQD
                relevantIndex = 3

            if fSplit[0][relevantIndex] == "x":
                if fSplit[0][-1] == "Q":
                    newfeature.insert(1, "Ant./PosteriorTranslation (norm.)")
                elif fSplit[0][-2:] == "QD":
                    newfeature.insert(1, "Ant./PosteriorNV (norm.)")
            if fSplit[0][relevantIndex] == "y":
                if fSplit[0][-1] == "Q":
                    newfeature.insert(1, "LateralTranslation (norm.)")
                elif fSplit[0][-2:] == "QD":
                    newfeature.insert(1, "LateralNV (norm.)")
            if fSplit[0][relevantIndex] == "z":
                if fSplit[0][-1] == "Q":
                    newfeature.insert(1, "VerticalTranslation (norm.)")
                elif fSplit[0][-2:] == "QD":
                    newfeature.insert(1, "VerticalNV (norm.)")

        else: # 18.03.2024:: Just for FootProgression
            # Angle or NAV
            if fSplit[0][-1] == "Q":
                newfeature.insert(1, "Angle\n")
            elif fSplit[0][-2:] == "QD":
                newfeature.insert(1, "NAV\n")

        # Phases
        if "Stride" in fSplit[1]:
            newfeature.insert(3, "(Stride")
        elif "Swing" in fSplit[1]:
            newfeature.insert(3, "(Swing")
        elif "Stance" in fSplit[1]:
            newfeature.insert(3, "(Stance")
        elif "Swg" in fSplit[1]:
            if "Pr" == fSplit[1][:2]:
                newfeature.insert(3, "(Pre-Swing")
            elif "In" == fSplit[1][:2]:
                newfeature.insert(3, "(Initial Swing")
            elif "Md" == fSplit[1][:2]:
                newfeature.insert(3, "(Mid-Swing")
            elif "Tr" == fSplit[1][:2]:
                newfeature.insert(3, "(Terminal Swing")
        elif "Stn" in fSplit[1]:
            if "Md" == fSplit[1][:2]:
                newfeature.insert(3, "(Midstance")
            elif "Tr" == fSplit[1][:2]:
                newfeature.insert(3, "(Terminal Stance")
        elif "LdRsp" in fSplit[1]:
            newfeature.insert(3, "(Loading Response")
        elif "InCntValue" in fSplit[1]:
            newfeature.insert(3, "(Initial Contact")

        # Min, Max, Median
        if "Min" in fSplit[1]:
            newfeature.insert(4, "min.)")
        elif "Max" in fSplit[1]:
            newfeature.insert(4, "max.)")
        elif "Median" in fSplit [1]:
            newfeature.insert(4, "median)")

    elif len(fSplit) == 3:
        # Phases
        if "Stride" in fSplit[0]:
            newfeature.insert(0, "Stride")
        elif "Swing" in fSplit[0]:
            newfeature.insert(0, "Swing")
        elif "Stance" in fSplit[0]:
            newfeature.insert(0, "Stance")
        elif "Swg" in fSplit[0]:
            if "Pr" == fSplit[0][:2]:
                newfeature.insert(0, "Pre-Swing")
            elif "In" == fSplit[0][n:2]:
                newfeature.insert(0, "Initial Swing")
            elif "Md" == fSplit[0][:2]:
                newfeature.insert(0, "Mid-Swing")
            elif "Tr" == fSplit[0][:2]:
                newfeature.insert(0, "Terminal Swing")
        elif "Stn" in fSplit[0]:
            if "Md" == fSplit[0][:2]:
                newfeature.insert(0, "Midstance")
            elif "Tr" == fSplit[0][:2]:
                newfeature.insert(0, "Terminal Stance")
        elif "LdRsp" in fSplit[0]:
            newfeature.insert(0, "Loading Response")

        # Start or Width
        if "Start" in fSplit[1]:
            newfeature.insert(1, "Start")
        elif "Width" in fSplit[1]:
            newfeature.insert(1, "Duration")

        # Unaffected and Affected
        if fSplit[2] == "UnAff":
            newfeature.insert(2, "contra.")
        elif fSplit[2] == "Aff":
            newfeature.insert(2, "ipsi.")

        if "Pelvis" in fSplit[0]:
            newfeature.insert(0, "Pelvis")
        elif "Spine" in fSplit[0]:
            newfeature.insert(0, "Spine")
        elif "Hip" in fSplit[0]:
            newfeature.insert(0, "Hip")
        elif "Knee" in fSplit[0]:
            newfeature.insert(0, "Knee")
        elif "Ankle" in fSplit[0]:
            newfeature.insert(0, "Ankle")
        elif "Thorax" in fSplit[0]:
            newfeature.insert(0, "Thorax")
        elif "Shoulder" in fSplit[0]:
            newfeature.insert(0, "Shoulder")
        elif "Elbow" in fSplit[0]:
            newfeature.insert(0, "Elbow")
        elif "Wrist" in fSplit[0]:
            newfeature.insert(0, "Wrist")

        # Angle Portion
        if "Dorsiflexion" in fSplit[1]:
            newfeature.insert(1, "Dorsiflexion")
        elif "Flexion" in fSplit[1]:
            newfeature.insert(1, "Flex./Ex.")
        elif "IntRotation" in fSplit[1]:
            newfeature.insert(1, "Rotation")
        elif "InvEversion" in fSplit[1]:
            newfeature.insert(1, "Inversion/Eversion")
        elif "SideTilt" in fSplit[1]:
            if "Pelvis" in fSplit[0]:
                newfeature.insert(1, "Obliquity")
            elif "Thorax" in fSplit[0]:
                newfeature.insert(1, "Side Tilt")
            elif "Spine" in fSplit[0]:
                newfeature.insert(1, "Side Tilt")

        elif "ForwardTilt" in fSplit[1]:
            newfeature.insert(1, "Tilt")
        elif "VarAdduction" in fSplit[1]:
            newfeature.insert(1, "Varus")
        elif "Adduction" in fSplit[1]:
            newfeature.insert(1, "Adduction")
        elif "Abduction" in fSplit[1]:
            newfeature.insert(1, "Abduction")
        elif "RadUlnarDev" in fSplit[1]:
            newfeature.insert(1, "Radius/Ulnar Deviation")

        # Angle or NAV
        if fSplit[1][-1] == "Q":
            newfeature.insert(2, "Angle\n")
        elif fSplit[1][-2:] == "QD":
            newfeature.insert(2, "NAV\n")

        # Unaffected and Affected
        if fSplit[0][:3] == "UnA":
            newfeature.insert(3, "contra.")
        elif fSplit[0][:1] == "A":
            newfeature.insert(3, "ipsi.")

        # Phases
        if "Stride" in fSplit[2]:
            newfeature.insert(4, "(Stride")
        elif "Swing" in fSplit[2]:
            newfeature.insert(4, "(Swing")
        elif "Stance" in fSplit[2]:
            newfeature.insert(4, "(Stance")
        elif "Swg" in fSplit[2]:
            if "Pr" == fSplit[2][:2]:
                newfeature.insert(4, "(Pre-Swing")
            elif "In" == fSplit[2][:2]:
                newfeature.insert(4, "(Initial Swing")
            elif "Md" == fSplit[2][:2]:
                newfeature.insert(4, "(Mid-Swing")
            elif "Tr" == fSplit[2][:2]:
                newfeature.insert(4, "(Terminal Swing")
        elif "Stn" in fSplit[2]:
            if "Md" == fSplit[2][:2]:
                newfeature.insert(4, "(Midstance")
            elif "Tr" == fSplit[2][:2]:
                newfeature.insert(4, "(Terminal Stance")
        elif "LdRsp" in fSplit[2]:
            newfeature.insert(4, "(Loading Response")
        elif "InCntValue" in fSplit[2]:
            newfeature.insert(4, "(Initial Contact")

        # Min, Max, Median, ROM
        if "Min" in fSplit[2]:
            newfeature.insert(5, "min.)")
        elif "Max" in fSplit[2]:
            newfeature.insert(5, "max.)")
        elif "Median" in fSplit[2]:
            newfeature.insert(5, "median)")
        elif "ROM" in fSplit[2]:
            newfeature.insert(5, "ROM)")
    
    newName = (' ').join(newfeature)

    return newName
