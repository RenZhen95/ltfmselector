import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os, sys
import random
import pickle
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed

from .env import Environment
from .utils import ReplayMemory, DQN, Transition, balance_classDistribution_patient

from itertools import count

from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Functions to clip predicted regression values
capUpperValues = lambda x: 3.0 if x > 3.0 else x
capLowerValues = lambda x: 0.0 if x < 0.0 else x

class LTFMSelectorVectorized:
    def __init__(
            self, episodes, batch_size=512, epochs=10, tau=0.0005,
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

        epochs : int
            Number of epochs in updating the policy network at each update
            step

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
            If `fQueryFunction == {'step', 'linear', 'quadratic'}`, upper
            limit of penalty

        fRate : None or float
            If `fQueryFunction == {'linear', 'quadratic'}`, rate of
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

            If it is intended for the agent to only dynamically select features
            and use only the prescribed prediction model, simply enter a list
            with one instance of a prediction model.

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
        self.epochs = epochs
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
        self.ReplayMemory = ReplayMemory(50000)

        # Initialize counter to track total actions already taken
        self.total_actions = 0

    def fit(
            self, X, y, loss_function='mse', sample_weight=None,
            agent_neuralnetwork=None, lr=1e-5, monitor=False,
            log_actions=False, background_dataset=None, **kwargs
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

        monitor : bool
            Monitor training process using a TensorBoard.

            Run `tensorboard --logdir=runs` in the terminal to monitor the
            progression of the action-value function.

        background_dataset : None or pd.DataFrame
            If None, numerical features will be assumed when computing the
            background dataset.

            The background dataset defines the feature values when a feature
            is not selected.

        log_actions : bool
            If `True`, the progression of selected actions will be saved in
            `self.ActionsLog` which is a np.array of size
            (self.episodes, self.max_timesteps).

            Because not every episode has the length defined in
            `self.max_timesteps`, remaining time-steps not "filled" will
            assume values of -2.
        '''
        ### Special-tailored implementation ###
        if "smsproject" in list(kwargs.keys()):
            self.smsproject = True
        else:
            self.smsproject = False

        # Training dataset
        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X)
        else:
            self.X = X

        if isinstance(y, np.ndarray):
            self.y = pd.Series(y)
        else:
            self.y = y

        # Compute background dataset if needed
        if background_dataset is None:
            # Computing background dataset (assuming numerical features)
            self.background_dataset = pd.DataFrame(
                data=np.zeros(self.X.shape), index=self.X.index,
                columns=self.X.columns
            )
            for i in self.background_dataset.index:
                self.background_dataset.loc[i] = self.X.drop(i).mean(axis=0)

            self.background_dataset.loc["Total", :] = self.X.mean(axis=0)
        else:
            self.background_dataset = background_dataset

        self.sample_weight = sample_weight

        # If user wants to monitor progression of terms in the loss function
        if monitor:
            writer = SummaryWriter()
            monitor_count = 1

        # If user wants to log actions
        if log_actions:
            self.ActionsLog = np.ones((self.episodes, self.max_timesteps))*(-2)

        # Getting background predictions here to generate multiple environments
        # in parallel
        self.y_pred_bg = self.get_bgPrediction(**kwargs)

        # Initialize probability of each feature sampled as the initial feature
        InitialFeatureP = self.get_probInitialFeature(self.X, self.y)

        # Initializing the environments in parallel
        envs = [
            Environment(
                self.X, self.y, self.background_dataset, self.y_pred_bg,
                self.fQueryCost, self.fQueryFunction,
                self.fThreshold, self.fCap, self.fRate,
                self.mQueryCost,
                self.fRepeatQueryCost, self.p_wNoFCost, self.errorCost,
                self.pType, InitialFeatureP,
                self.regression_tol, self.regression_error_rounding,
                self.pModels, self.device, sample_weight=self.sample_weight, **kwargs
            ) for i in tqdm(range(self.episodes), desc=f"Creating {self.episodes} environments")
        ]
        if log_actions:
            IDX = [i for i in range(self.episodes)]

        # Take an environment and reset it simply for metadata initialization
        _intenv = Environment(
            self.X, self.y, self.background_dataset, self.y_pred_bg,
            self.fQueryCost, self.fQueryFunction,
            self.fThreshold, self.fCap, self.fRate,
            self.mQueryCost,
            self.fRepeatQueryCost, self.p_wNoFCost, self.errorCost,
            self.pType, InitialFeatureP,
            self.regression_tol, self.regression_error_rounding,
            self.pModels, self.device, sample_weight=self.sample_weight
        )
        _intenv.reset()

        self.state_length   = len(_intenv.state)
        self.actions_length = len(_intenv.actions)

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
                len(_intenv.state), len(_intenv.actions), nLayer1, nLayer2
            ).to(self.device)

            self.target_net = DQN(
                len(_intenv.state), len(_intenv.actions), nLayer1, nLayer2
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initializing the optimizer
        optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

        # Training the agent over self.episodes
        if self.max_timesteps is None:
            self.max_timesteps = _intenv.nFeatures * 3

        # Reset all environments
        states = np.array(Parallel(n_jobs=-1, prefer="threads")(
            delayed(env.reset)() for env in envs
        ))
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        # >> Tensor(nEpisodes, |State|)

        for t in count():
            actions = self.select_action(states, envs)
            # >> Tensor(nEpisodes, 1) - Action selected for each environment

            # Log actions if desired
            if log_actions:
                self.ActionsLog[IDX, t] = actions.squeeze(1).numpy()

            # If maximum time_steps is reached
            if t+1 == self.max_timesteps:
                actions = torch.tensor(
                    [-1 for e in range(len(envs))], device=self.device
                ).view(len(envs), 1)

            # Agent carries out action in each environment and returns:
            # - Observations :: list(nEpisodes)
            # >> Every element is the next state or None (termination)
            # - Rewards      :: list(nEpisodes)
            # - Terminations :: list(nEpisodes of boolean value)
            envTransition = []
            for envIdx, env in enumerate(tqdm(envs)):
                envTransition.append(env.step(actions[envIdx, 0].item()))

            observations, rewards, terminations = zip(*envTransition)
            observations = list(observations)
            rewards = torch.tensor(
                list(rewards), device=self.device
            ).view(len(envs), 1)
            terminations = list(terminations)

            get_nextState = lambda x, o: None if x else torch.tensor(
                o, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            nextStates = Parallel(n_jobs=-1, prefer="threads")(
                delayed(get_nextState)(t, o) for t, o in zip(terminations, observations)
            )

            # Push to replay buffer
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.ReplayMemory.push)(
                    s.unsqueeze(0), a.unsqueeze(0), sp, r
                ) for s, a, sp, r in zip(
                    torch.unbind(states, dim=0), torch.unbind(actions, dim=0),
                    nextStates, rewards
                )
            )

            # Update next state
            # - Kick out environments that have ended from `envs` and `states`
            nextStates_onGoing = []
            envIdxToRemove     = []
            for i, sp in enumerate(nextStates):
                if not sp is None:
                    nextStates_onGoing.append(sp)
                else:
                    envIdxToRemove.append(i)

            for i in reversed(envIdxToRemove):
                del envs[i]
                if log_actions:
                    del IDX[i]

            # All environments have terminated
            if len(nextStates_onGoing) == 0:
                break

            states = torch.tensor(
                np.array(nextStates_onGoing), dtype=torch.float32, device=self.device
            ).squeeze(1)

            # Optimize the model over user-desired number of epochs
            for _ in range(self.epochs):
                _res = self.optimize_model(optimizer, loss_function, monitor)
            sys.exit()
            if monitor:
                writer.add_scalar("Metrics/Average_QValue", _res[0], monitor_count)
                writer.add_scalar("Metrics/Average_Reward", _res[1], monitor_count)
                writer.add_scalar("Metrics/Average_Target", _res[2], monitor_count)
                monitor_count += 1

            # Apply soft update to target network's weights
            targetParameters = self.target_net.state_dict()
            policyParameters = self.policy_net.state_dict()

            for key in policyParameters:
                targetParameters[key] = policyParameters[key]*self.tau + \
                    targetParameters[key]*(1 - self.tau)

            self.target_net.load_state_dict(targetParameters)

            # Saving trained policy network intermediately
            if not self.checkpoint_interval is None:
                if (i_episode + 1) % self.checkpoint_interval == 0:
                    self.policy_network_checkpoints[i_episode + 1] =\
                        self.policy_net.state_dict()

        # Save trained weights after all episodes
        self.policy_network_checkpoints[self.episodes] = self.policy_net.state_dict()

        if monitor:
            writer.add_scalar("Metrics/Average_QValue", _res[0], monitor_count)
            writer.add_scalar("Metrics/Average_Reward", _res[1], monitor_count)
            writer.add_scalar("Metrics/Average_Target", _res[2], monitor_count)
            writer.flush()
            writer.close()

    def predict(self, X_test, log=False, **kwargs):
        '''
        Use trained agent to select features and a suitable prediction model
        to predict the target/class, given X_test.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test samples

        log : bool
            Log each predicted sample into self.doc_test

        Returns
        -------
        y_pred : pd.Series
            Target/Class predicted for X_test
        '''
        # Initializing the environments in parallel for each test sample
        envs = [
            Environment(
                self.X, self.y, self.background_dataset, self.y_pred_bg,
                self.fQueryCost, self.fQueryFunction,
                self.fThreshold, self.fCap, self.fRate,
                self.mQueryCost,
                self.fRepeatQueryCost, self.p_wNoFCost, self.errorCost,
                self.pType, self.regression_tol, self.regression_error_rounding,
                self.pModels, self.device, sample_weight=self.sample_weight,
                **kwargs
            ) for i in tqdm(
                range(X_test.shape[0]),
                desc=f"Creating {X_test.shape[0]} environments for each test sample"
            )
        ]
        if isinstance(X_test, pd.DataFrame):
            IDX = list(X_test.index)
        else:
            raise ValueError("X_test must be a pandas DataFrame!")

        # Create dictionary to save information per episode
        if log:
            self.doc_test = defaultdict(dict)

        # Array to store predictions
        y_pred = pd.Series(np.zeros(X_test.shape[0]), index=X_test.index)

        # Reset all environments
        states = np.array(Parallel(n_jobs=-1, prefer="threads")(
            delayed(env.reset)(
                sample=X_test.iloc[[i]]
            ) for i, env in enumerate(envs)
        ))
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        for t in count():
            selector = ptan.actions.EpsilonGreedyActionSelector(
                epsilon=self.getEpsilon(t)
            )
            # actions = self.select_action(states, envs)

            if t+1 == self.max_timesteps:
                actions = torch.tensor(
                    [-1 for e in range(len(envs))], device=self.device
                ).view(len(envs), 1)

            envTransition = []
            for envIdx, env in enumerate(tqdm(envs)):
                envTransition.append(env.step(actions[envIdx, 0].item()))

            observations, rewards, terminations = zip(*envTransition)
            observations = list(observations)
            rewards = torch.tensor(
                list(rewards), device=self.device
            ).view(len(envs), 1)
            terminations = list(terminations)

            get_nextState = lambda x, o: None if x else torch.tensor(
                o, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            nextStates = Parallel(n_jobs=-1, prefer="threads")(
                delayed(get_nextState)(t, o) for t, o in zip(terminations, observations)
            )

            # Update next state
            # - Kick out environments that have ended from `envs` and `states`
            nextStates_onGoing = []
            envIdxToRemove     = []
            for i, sp in enumerate(nextStates):
                if not sp is None:
                    nextStates_onGoing.append(sp)
                else:
                    envIdxToRemove.append(i)
                    y_pred.at[IDX[i]] = envs[i].y_pred

                    if log:
                        doc_episode = {
                            "SampleID": IDX[i],
                            "PredModel": envs[i].get_prediction_model(),
                            "Iterations": t+1,
                            "Mask": envs[i].get_feature_mask(),
                            "predModel_nChanges": envs[i].pm_nChange
                        }
                        self.doc_test[IDX[i]] = doc_episode

            for i in reversed(envIdxToRemove):
                del envs[i]
                del IDX[i]

            # All environments have terminated
            if len(nextStates_onGoing) == 0:
                break

            states = torch.tensor(
                np.array(nextStates_onGoing), dtype=torch.float32, device=self.device
            ).squeeze(1)

        return y_pred

    def select_action(self, states, envs):
        '''
        Select an action based on the given states. For exploration an
        epsilon-greedy strategy is implemented - the agent will for an
        epsilon probability choose a random action, instead of using the
        policy network.

        Parameters
        ----------
        states : torch.Tensor
            State of environments generated in parallel
        '''
        # Probability of choosing random actions, instead of best action
        # - Probability decreases exponentially over time
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.total_actions / self.eps_decay)

        self.total_actions += 1

        # Perform random action
        if eps_threshold > random.random():
            return torch.tensor(
                [env.get_random_action() for env in envs], device=self.device, dtype=torch.long
            ).view(states.shape[0], 1)

        # Perform maximizing action
        else:
            with torch.no_grad():
                QValues = self.policy_net(states)
                # >> Tensor(nEpisodes, |Actions (Predict + #Features + #PM)|)

                return (QValues.max(1)[1].view(states.shape[0], 1) - 1)

    def optimize_model(self, optimizer, loss_function, monitor):
        '''
        Optimize the policy network.

        Parameters
        ----------
        loss_function : {'mse', 'smoothl1'} or custom function
            Choice of loss function. Default is 'mse'. User may also pass
            own customized loss function, based on PyTorch.
        '''
        # Regarding notations used in comments:
        # s  : current state
        # a  : action
        # s' : future state
        # Q  : action-value function (quality)
        #      (estimate of the cumulative reward, R)

        if len(self.ReplayMemory) < self.batch_size:
            if monitor:
                _res = (0., 0., 0.)
            else:
                _res = None

            return _res

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
        print(state_batch)
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

        # Step ---
        # Double Deep Q-Learning
        # 6. Compute r + GAMMA * {Qt(s', argmax_a Q(s', a))}

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

        # Clip target values at a maximum of zero, since we only have penalties!
        expected_state_action_values = torch.clamp(
            expected_state_action_values, max=0.0
        )

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

        if monitor:
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

    def get_bgPrediction(self, **kwargs):
        '''
        Get prediction based on average targets/classes when the agent decides
        to make a prediction without any recruited features.
        '''
        if isinstance(self.y, pd.Series):
            _y = (self.y.values).copy()
        else:
            _y = self.y.copy()

        if self.smsproject:
            X_wTarget = self.X.iloc[:,0:2].copy()
            X_wTarget["Target"] = _y
            X_wTarget = X_wTarget.rename(
                index=lambda x: x[0:5] if x.startswith('ES') else x[0:8]
            )
            X_wTarget_patLevel = X_wTarget.groupby("StridePairID").mean()
            _y = X_wTarget_patLevel["Target"]

        return _y.mean()

    def get_probInitialFeature(self, X, y):
        '''
        Trains a Random Forest ensemble of 10000 trees, calculates feature
        importance, and returns the probability of sampling a feature, based
        on its total relative relevance.

        Returns
        -------
        X : pd.DataFrame
            Training dataset, pandas dataframe with the shape:
            (n_samples, n_features)

        y : pd.Series
            Class/Target vector

        Returns
        -------
        probs : numpy.ndarray
         - Probabily of sampling feature based on their relevance
        '''
        print(
            "Initializing the probability of a feature being selected as " +
            "an initial feature, based on the feature's importance in a " +
            "random forest ensemble ..."
        )
        if self.pType == "regression":
            rf = RandomForestRegressor(
                n_estimators=100, n_jobs=-1, random_state=42
            )
        elif self.pType == "classification":
            rf = RandomForestClassifier(
                n_estimators=10000, n_jobs=-1, random_state=42
            )
        else:
            raise ValueError("'pType' must be 'regression' or 'classification'")

        rf.fit(X, y)

        # Get feature importance
        # > scikit-learn's feature_importances_ sum to 1.0 by default
        probs = rf.feature_importances_

        return probs
