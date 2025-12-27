import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import random
import pickle
import tarfile
import numpy as np
import pandas as pd
from collections import defaultdict

from .env import Environment
from .utils import ReplayMemory, DQN, Transition

from itertools import count

from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def load_model(filename, default_policy_net=True):
    '''
    Load a previously saved model. The list of prediction models will be
    returned. User should manually load it, see example. This design
    decision was made to account for users who might work with PyTorch and
    TensorFlow models which should not be pickled whole.

    Parameter
    ---------
    filename : str or Path.pathlib

    default_policy_net : bool
        If True, user used the default policy network during training

    Returns
    -------
    pModels : list(tuple)
        List of prediction models provided by user during training
    '''
    with tarfile.open(filename, 'r:gz') as tar:
        with tar.extractfile('selector.pkl_ltfmselector') as f:
            selector = pickle.load(f)

        with tar.extractfile('pModels.pkl_list') as f:
            pModels = pickle.load(f)

        with tar.extractfile('policy_network_checkpoints.pkl_dict') as f:
            policy_network_dict = pickle.load(f)

    # Load policy network
    if default_policy_net:
        policy_network = DQN(
            selector.state_length, selector.actions_length,
            policy_network_dict["n1"],
            policy_network_dict["n2"]
        )
        policy_network.load_state_dict(
            policy_network_dict[selector.episodes]
        )
    selector.policy_net = policy_network

    # Set pModels to None to ENSURE that user, manually sets the prediction
    # models used earlier.
    selector.pModels = None

    return selector, pModels

class LTFMSelector:
    def __init__(
            self, episodes, batch_size=256, tau=0.0005,
            eps_start=0.9, eps_end=0.05, eps_decay=1000,
            fQueryCost=0.01, fQueryFunction=None,
            fThreshold=None, fCap=None, fRate=None,
            mQueryCost=0.01,
            fRepeatQueryCost=1.0, p_wNoFCost=5.0, errorCost=1.0,
            pType="regression", regression_tol=0.5,
            regression_error_rounding=1,
            pModels=None,
            gamma=0.99, max_timesteps=None,
            checkpoint_interval=None, device="cpu"
    ):
        '''
        Locally-Tailored Feature and Model Selector, implemented according
        to the method described in https://doi.org/10.17185/duepublico/82941.

        Parameters
        ----------
        episodes : int
            Number of episodes agent is trained

        batch_size : int
            Batch size to train the policy network with

        tau : float
            Update rate of the target network

        eps_start : float
            Start value of epsilon

        eps_end : float
            Final value of epsilon

        eps_decay : float
            Rate of exponential decay

        fQueryCost : float
            Cost of querying a feature.

        fQueryFunction : None or {'step', 'linear', 'quadratic'}
            User can also decide to progressively increase the cost of 
            querying features in the following manner:
            'step' : 
                Every additional feature adds a fixed constant, determined 
                by user.
            'linear' : 
                Cost of every additional feature linearly increases according    
                to user-defined gradient
            'quadratic' : 
                Cost of every additional feature increases quadratically, 
                according to a user-defined rate

        fThreshold : None or int
            If `fQueryFunction == {'step', 'linear', 'quadratic', 'exponential'}`
            Threshold of number of features, before cost of recruiting 
            increases
            
        fCap : None or float
            If `fQueryFunction == {'step'}`, upper limit of penalty

        fRate : None or float
            If `fQueryFunction == {'linear', 'quadratic', 'exponential'}`, rate
            individual cost functions

        mQueryCost : float
            Cost of querying a prediction model

        fRepeatQueryCost : float
            Cost of querying a feature already previously selected

        p_wNoFCost : float
            Cost of switching selected prediction model

        errorCost : float
            Cost of making a wrong prediction

            If pType == 'regression', then
            Agent is punished -errorCost*abs(``prediction`` - ``target``)

            If pType == 'classification', then
            Agent is punished -errorCost

        pType : {'regression' or 'classification'}
            Type of prediction to make

        regression_tol : float
            Only applicable for regression models, punish agent if prediction
            error is bigger than regression_tol

        regression_error_rounding : int (default = 1)
            Only applicable for regression models. The error between the
            prediction and true value is rounded to the input decimal place.

        pModels : None or ``list of prediction models``
            Options of prediction models that the agent can choose from

            If None, the default options will include for classification:
            1. Support Vector Machine
            2. Random Forest
            3. Gaussian Naive Bayes

            For regression:
            1. Support Vector Machine
            2. Random Forest
            3. Ridge Regression

        gamma : float
            Discount factor, must be in :math:`]0, 1]`. The higher the discount
            factor, the higher the influence of rewards from future states.

            In other words, the more emphasis is placed on maximizing rewards
            with a long-term perspective. A discount factor of zero would result
            in an agent that only seeks to maximize immediate rewards.

        max_timesteps : int or None
            Maximum number of time-steps per episode. Agent will be forced to
            make a prediction with the selected features and prediction model,
            if max_timesteps is reached

            If None, max_timesteps will be set to 3 x number_of_features

        checkpoint_interval : int or None
            Save the policy network after a defined interval of episodes as
            checkpoints. Obviously cannot be more than ``episodes``
        '''
        self.device = device

        self.batch_size = batch_size
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.episodes = episodes
        self.max_timesteps = max_timesteps
        self.checkpoint_interval = checkpoint_interval
        self.policy_net = None
        self.policy_network_checkpoints = dict()

        if not checkpoint_interval is None:
            if checkpoint_interval > max_timesteps:
                raise ValueError(
                    "Invalid value for 'checkpoint_interval', it must be " +
                    "less than 'max_timesteps'!"
                )

        if not pType in ["regression", "classification"]:
            raise ValueError("Either 'regression' or 'classification' only!")
        else:
            self.pType = pType

        # Reward function
        self.fQueryCost = fQueryCost
        self.fQueryFunction = fQueryFunction
        self.fThreshold = fThreshold
        self.fCap = fCap
        self.fRate = fRate

        # Options for progressive cost functions
        if isinstance(self.fQueryFunction, str):
            fQueryFunctions = ['step', 'linear', 'quadratic']
            if not self.fQueryFunction in fQueryFunctions:
                raise ValueError(
                    f"{self.fQueryFunction} is not a valid option. Available " +
                    f"options are {fQueryFunctions}"
                )
            else:
                if not isinstance(fThreshold, int):
                    raise ValueError("Parameter fThreshold must be an integer!")

                if self.fQueryFunction == "step":
                    if not (isinstance(fCap, float) or isinstance(fCap, int)):
                        raise ValueError("Parameter fCap must be an int or float!")
                    else:
                        self.fCap = float(fCap)
                else:
                    if self.fQueryFunction in ["linear", "quadratic"]:
                        if not (isinstance(fRate, float) or isinstance(fRate, int)):
                            raise ValueError("Parameter fRate must be an int or float!")
                        else:
                            self.fRate = float(fRate)

        self.mQueryCost = mQueryCost
        self.fRepeatQueryCost = fRepeatQueryCost
        self.p_wNoFCost = p_wNoFCost
        self.errorCost = errorCost
        self.regression_tol = regression_tol
        self.regression_error_rounding = regression_error_rounding

        # Available option of prediction models the agent can select
        if (pModels is None) and (self.pType == "regression"):
            self.pModels = [
                SVR(),
                RandomForestRegressor(n_jobs=-1),
                Ridge()
            ]
        elif (pModels is None) and (self.pType == "classification"):
            self.pModels = [
                SVC(),
                RandomForestClassifier(n_jobs=-1),
                GaussianNB()
            ]
        else:
            self.pModels = pModels

        # Initializing the ReplayMemory
        self.ReplayMemory = ReplayMemory(10000)

        self.total_actions = 0

    def fit(
            self, X, y, loss_function='mse', sample_weight=None,
            agent_neuralnetwork=None, lr=1e-5, returnQ=False,
            monitor=False, background_dataset=None, **kwargs
    ):
        '''
        Initializes the environment and agent, then trains the agent to select
        optimal combinations of features and prediction models locally, i.e.
        specific for a given sample.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe with the shape: (n_samples, n_features)

        y : pd.Series
            Class/Target vector

        loss_function : {'mse', 'smoothl1'} or custom function
            Choice of loss function. Default is 'mse'. User may also pass
            own customized loss function, based on PyTorch.

        sample_weight : list or array or None
            Per-sample weights

        agent_neuralnetwork : torch.nn.Module or (int, int) or None
            Neural network to represent the policy network of the agent.

            User may pass user-defined PyTorch neural network or a tuple of two
            integer elements (n1, n2). n1 and n2 pertains to the number of units
            in the first and second layer of a multilayer-perceptron,
            implemented in PyTorch.

            If None, a default multilayer-perceptron of two hidden layers, each
            with 1024 units is used.

        lr : float
            Learning rate of the default AdamW optimizer to optimize parameters
            of the policy network

        returnQ : bool
            Return average computed action-value functions and rewards of
            the sampled batches, as a (<total_iterations>, 3) matrix. The 
            columns correspond to the averaged Q, reward, and target functions.

        monitor : bool
            Monitor training process using a TensorBoard.

            Run `tensorboard --logdir=runs` in the terminal to monitor the p
            progression of the action-value function.

        background_dataset : None or pd.DataFrame
            If None, numerical features will be assumed when computing the
            background dataset.

            The background dataset defines the feature values when a feature
            is not selected.

        Returns
        -------
        doc : dict
            Log/documentation of each training episode

        action_values_Q : tuple
            Q_avr_list : list
                List of policy network's action-value function, Q(s,a),
                averaged over the sampled batch during training, per iteration
            r_avr_list : list
                List of rewards, r, averaged over the sampled batch during
                training, per iteration
            V_avr_list : list
                List of max action-value function for the next state (s'),
                max{a} Q(s', a), averaged over the sampled batch during
                training, per iteration
        '''
        # Training dataset
        self.X = X
        self.y = y

        # Compute background dataset if needed
        if background_dataset is None:
            # Computing background dataset (assuming numerical features)
            self.background_dataset = pd.DataFrame(
                data=np.zeros(X.shape), index=X.index,
                columns=X.columns
            )
            for i in self.background_dataset.index:
                self.background_dataset.loc[i] = X.drop(i).mean(axis=0)

            self.background_dataset.loc["Total", :] = X.mean(axis=0)
        else:
            self.background_dataset = background_dataset

        self.sample_weight = sample_weight

        # If user wants to monitor progression of terms in the loss function
        if monitor:
            writer = SummaryWriter()
            monitor_count = 1

        # If user wants to save average computed action-value functions and
        # rewards of sampled batches
        if returnQ:
            total_iterations = 16777216 # 2^24
            LearningValuesMatrix = np.zeros(
                (total_iterations, 3), dtype=np.float32
            )
            Q_count = 1

        # Initializing the environment
        env = Environment(
            self.X, self.y, self.background_dataset,
            self.fQueryCost, self.fQueryFunction,
            self.fThreshold, self.fCap, self.fRate,
            self.mQueryCost,
            self.fRepeatQueryCost, self.p_wNoFCost, self.errorCost,
            self.pType, self.regression_tol, self.regression_error_rounding,
            self.pModels, self.device, sample_weight=self.sample_weight,
            **kwargs
        )
        env.reset()

        # Initializing length of state and actions as public fields for
        # loading the model later
        self.state_length = len(env.state)
        self.actions_length = len(env.actions)

        # Initializing the policy and target networks
        if isinstance(agent_neuralnetwork, nn.Module):
            self.policy_net = agent_neuralnetwork
            self.target_net = agent_neuralnetwork

        else:
            if agent_neuralnetwork is None:
                nLayer1 = 1024
                nLayer2 = 1024

            elif isinstance(agent_neuralnetwork, tuple) and len(agent_neuralnetwork) == 2:
                nLayer1 = agent_neuralnetwork[0]
                nLayer2 = agent_neuralnetwork[1]

            self.policy_net = DQN(
                len(env.state), len(env.actions), nLayer1, nLayer2
            ).to(self.device)

            self.target_net = DQN(
                len(env.state), len(env.actions), nLayer1, nLayer2
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initializing the optimizer
        optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

        # Create dictionary to save information per episode
        doc = defaultdict(dict)

        # Training the agent over self.episodes
        if self.max_timesteps is None:
            self.max_timesteps = env.nFeatures * 3

        for i_episode in range(self.episodes):
            print(f"\n\n=== Episode {i_episode+1} === === ===")
            state = env.reset()

            # Convert state to pytorch tensor
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            for t in count():
                # Make agent take an action
                action = self.select_action(state, env)

                if t > self.max_timesteps:
                    action = torch.tensor([[-1]], device=self.device)

                # Agent carries out action on the environment and returns:
                # - observation (state in next time-step)
                # - reward
                observation, reward, terminated = env.step(action.item())

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                self.ReplayMemory.push(
                    state, action, next_state,
                    torch.tensor([reward], device=self.device)
                )

                # Move on to next state
                state = next_state

                # Optimize the model
                _res = self.optimize_model(optimizer, loss_function, monitor, returnQ)

                if monitor:
                    if not _res is None:
                        writer.add_scalar("Metrics/Average_QValue", _res[0], monitor_count)
                        writer.add_scalar("Metrics/Average_Reward", _res[1], monitor_count)
                        writer.add_scalar("Metrics/Average_Target", _res[2], monitor_count)
                        monitor_count += 1

                if returnQ:
                    if not _res is None:
                        LearningValuesMatrix[Q_count, 0] = _res[0]
                        LearningValuesMatrix[Q_count, 1] = _res[1]
                        LearningValuesMatrix[Q_count, 2] = _res[2]
                        Q_count += 1

                # Apply soft update to target network's weights
                targetParameters = self.target_net.state_dict()
                policyParameters = self.policy_net.state_dict()

                for key in policyParameters:
                    targetParameters[key] = policyParameters[key]*self.tau + \
                        targetParameters[key]*(1 - self.tau)

                self.target_net.load_state_dict(targetParameters)

                if terminated:
                    doc_episode = {
                        "SampleID": env.X_test.index[0],
                        "y_true": env.y_test,
                        "y_pred": env.y_pred,
                        "PredModel": env.get_prediction_model(),
                        "Episode": i_episode + 1,
                        "Iterations": t+1,
                        "Mask": env.get_feature_mask(),
                        "predModel_nChanges": env.pm_nChange
                    }
                    doc[i_episode] = doc_episode

                    print("Episode terminated:")
                    print(
                        f"- Iterations                 : {doc_episode['Iterations']}\n" +
                        f"- Features selected          : {doc_episode['Mask'].sum()}\n" +
                        f"- Prediction model           : {doc_episode['PredModel']}\n" +
                        f"- Prediction model #(change) : {doc_episode['predModel_nChanges']}"
                    )
                    break

            # Saving trained policy network intermediately
            if not self.checkpoint_interval is None:
                if (i_episode + 1) % self.checkpoint_interval == 0:
                    self.policy_network_checkpoints[i_episode + 1] =\
                        self.policy_net.state_dict()

            if not self.episodes in self.policy_network_checkpoints:
                self.policy_network_checkpoints[self.episodes] =\
                    self.policy_net.state_dict()

        if monitor:
            writer.add_scalar("Metrics/Average_QValue", _res[0], monitor_count)
            writer.add_scalar("Metrics/Average_Reward", _res[1], monitor_count)
            writer.add_scalar("Metrics/Average_Target", _res[2], monitor_count)
            writer.flush()
            writer.close()

        if returnQ:
            LearningValuesMatrix[Q_count, 0] = _res[0]
            LearningValuesMatrix[Q_count, 1] = _res[1]
            LearningValuesMatrix[Q_count, 2] = _res[2]

        if (monitor or returnQ):
            return doc, LearningValuesMatrix[0:Q_count+1, :]
        else:
            return doc

    def predict(self, X_test, **kwargs):
        '''
        Use trained agent to select features and a suitable prediction model
        to predict the target/class, given X_test.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test samples

        Returns
        -------
        y_pred : array
            Target/Class predicted for X_test

        doc_test : dict
            Log/documentation of each test sample
        '''
        # Initializing the environment
        env = Environment(
            self.X, self.y, self.background_dataset,
            self.fQueryCost, self.fQueryFunction,
            self.fThreshold, self.fCap, self.fRate,
            self.mQueryCost,
            self.fRepeatQueryCost, self.p_wNoFCost, self.errorCost,
            self.pType, self.regression_tol, self.regression_error_rounding,
            self.pModels, self.device, **kwargs
        )

        # Create dictionary to save information per episode
        doc_test = defaultdict(dict)

        # Array to store predictions
        y_pred = np.zeros(X_test.shape[0])

        for i, test_sample in enumerate(X_test.index):
            print(f"\n\n=== Test sample {test_sample} === === ===")
            state = env.reset(sample=X_test.loc[[test_sample]])

            # Convert state to pytorch tensor
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            for t in count():
                action = self.select_action(state, env)

                if t > self.max_timesteps:
                    action = torch.tensor([[-1]], device=self.device)

                observation, reward, terminated = env.step(action.item())

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                state = next_state

                if terminated:
                    doc_episode = {
                        "SampleID": test_sample,
                        "PredModel": env.get_prediction_model(),
                        "Iterations": t+1,
                        "Mask": env.get_feature_mask(),
                        "predModel_nChanges": env.pm_nChange
                    }
                    doc_test[test_sample] = doc_episode

                    print("Episode terminated:")
                    print(
                        f"- Iterations                 : {doc_episode['Iterations']}\n" +
                        f"- Features selected          : {doc_episode['Mask'].sum()}\n" +
                        f"- Prediction model           : {doc_episode['PredModel']}\n" +
                        f"- Prediction model #(change) : {doc_episode['predModel_nChanges']}"
                    )
                    y_pred[i] = env.y_pred
                    break

        return y_pred, doc_test

    def select_action(self, state, env):
        '''
        Select an action based on the given state. For exploration an
        epsilon-greedy strategy is implemented - the agent will for an
        epsilon probability choose a random action, instead of using the
        policy network.

        Parameters
        ----------
        state : np.array
            State of environment
        '''
        # Probability of choosing random actions, instead of best action
        # - Probability decreases exponentially over time
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.total_actions / self.eps_decay)

        self.total_actions += 1

        if eps_threshold > random.random():
            return torch.tensor(
                [[env.get_random_action()]], device=self.device, dtype=torch.long
            )
        else:
            with torch.no_grad():
                return (self.policy_net(state).max(1)[1].view(1, 1) - 1)

    def optimize_model(self, optimizer, loss_function, monitor, returnQ):
        '''
        Optimize the policy network.

        Parameters
        ----------
        loss_function : {'mse', 'smoothl1'} or custom function
            Choice of loss function. Default is 'mse'. User may also pass
            own customized loss function, based on PyTorch.

        returnQ : bool
            Return average computed action-value functions and rewards of
            the sampled batches, for debugging purposes.
        '''
        # Regarding notations used in comments:
        # s  : current state
        # a  : action
        # s' : future state
        # Q  : action-value function (quality)
        #      (estimate of the cumulative reward, R)

        if len(self.ReplayMemory) < self.batch_size:
            return

        # Step ---
        # 1. Draw a random batch of experiences
        experiences = self.ReplayMemory.sample(self.batch_size)
        # [
        #    Experience #1: (state, action, next_state, reward),
        #    Experience #2: (state, action, next_state, reward),
        #    ...
        # ]

        # Step ---
        # 2. Convert the experiences into batches, per "item"
        batch = Transition(*zip(*experiences))
        # [
        #    s  : (#1, #2, ..., #BATCH_SIZE),
        #    a  : (#1, #2, ..., #BATCH_SIZE),
        #    s' : (#1, #2, ..., #BATCH_SIZE),
        #    r  : (#1, #2, ..., #BATCH_SIZE)
        # ]

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Step ---
        # 3. Get a boolean mask of non-final states (iterations)
        #    - s' is None if environment terminates
        non_final_mask = torch.tensor(
            tuple(
                map(lambda s: s is not None, batch.next_state)
            ), device=self.device, dtype=torch.bool
        )
        # Example of map()
        # >> A = [6, 53, 3, 9, 12]
        # >> B = tuple(map(lambda s: s < 10, A))
        # (True False True True False)

        # Step ---
        # 4. Get a batch of non-final next_states of tensor dimensions:
        #    - (<#BATCH_SIZE (except final states), (#features * 2)+1)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # Step ---
        # 5. Compute Q(s, a) of each sampled state-action pair from
        #    with the policy network
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch+1
        ).float()
        # action_batch+1 because the actions begin from [-1 0 1 2 ...],
        # where -1 indicates the action of making a prediction.

        # To get the Q(s,a) of a taken, add 1 to a-value to get the index
        # of the self.policy_net(state_batch) matrix, that pertains to the
        # selected action, a

        # Example: 3rd row of self.policy_net(state_batch) pertains to Q(s,a)
        # of selecting the second feature

        # Step ---
        # 6. Compute r + GAMMA * max_(a) {Q(s', a)} with the target network

        # Q(s', a) computed based on "older" target network, selecting for
        # action that maximizes this term

        # This is merged, per non_final_mask, such that we'll have either:
        #  1. r + GAMMA * max_(a) {Q(s', a)}
        #  2. 0 (cause that state was final for that episode)
        next_state_values = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0].float()

        expected_state_action_values = (
            reward_batch + (next_state_values * self.gamma)
        ).float()

        # Step ---
        # 7. Compute loss
        if isinstance(loss_function, str):
            if loss_function == 'mse':
                criterion = nn.MSELoss()
            elif loss_function == 'smoothl1':
                criterion = nn.SmoothL1Loss()
        else:
            criterion = loss_function

        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        optimizer.zero_grad()

        # Compute gradient via backpropagation
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        # Optimize the model (policy network)
        optimizer.step()

        if (monitor or returnQ):
            Q_avr = state_action_values.detach().numpy().mean()
            r_avr = reward_batch.unsqueeze(1).numpy().mean()
            V_avr = expected_state_action_values.unsqueeze(1).numpy().mean()
            res = (Q_avr, r_avr, V_avr)
        else:
            res = None

        return res

    def save_model(self, filename):
        '''
        Save the model. The LTFMSelector object will be pickled, but the
        prediction models (pModels) and the policy network (policy_net),
        will be saved separately.

        Parameters
        ----------
        filename : str
        '''
        # 1. Save the LTFMSelector object
        with open('selector.pkl_ltfmselector', 'wb') as f:
            pickle.dump(self, f)

        # 2. Save the prediction models
        with open('pModels.pkl_list', 'wb') as f:
            pModels_to_save = []

            for model in self.pModels:
                if isinstance(model, nn.Module):
                    pModels_to_save.append("pytorch", model.state_dict())
                else:
                    pModels_to_save.append((type(model), model))

            pickle.dump(pModels_to_save, f)

        # 3. Save the weights of the policy network
        with open('policy_network_checkpoints.pkl_dict', 'wb') as f:
            self.policy_network_checkpoints["n1"] = self.policy_net.n1
            self.policy_network_checkpoints["n2"] = self.policy_net.n2
            pickle.dump(self.policy_network_checkpoints, f)

        # 4. Save all in a tarball
        with tarfile.open(f"{filename}.tar.gz", 'w:gz') as tar:
            tar.add('selector.pkl_ltfmselector')
            tar.add('pModels.pkl_list')
            tar.add('policy_network_checkpoints.pkl_dict')

        os.remove('selector.pkl_ltfmselector')
        os.remove('pModels.pkl_list')
        os.remove('policy_network_checkpoints.pkl_dict')

    def __getstate__(self):
        state = self.__dict__.copy()

        del state["pModels"]
        del state["policy_net"]
        del state["target_net"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
