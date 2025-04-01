import random
import pandas as pd
from operator import attrgetter
from collections import deque, namedtuple, defaultdict

import torch.nn as nn
import torch.nn.functional as F

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
        self.layer1 = nn.Linear(n_observations, n1)
        self.layer2 = nn.Linear(n1, n2)
        self.layer3 = nn.Linear(n2, n_actions)

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
     - DataFrame extracted to be used as a template, mainly for the sample indices and their
       corresponding scores

    score : str
     - Medical score to target

    Returns
    -------
    df_sampleWeights : pd.DataFrame
     - DataFrame mapping each sample to its corresponding weight
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

