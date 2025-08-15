import torch
import torch.nn as nn
import torch.optim as optim

import random
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

class LTFMSelector:
    def __init__(
            self, episodes, batch_size=256, tau=0.0005,
            eps_start=0.9, eps_end=0.05, eps_decay=1000,
            fQueryCost=0.01, mQueryCost=0.01,
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
            Cost of querying a feature

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

        if not checkpoint_interval is None:
            if checkpoint_interval > max_timesteps:
                raise ValueError(
                    "Invalid value for 'checkpoint_interval', it must be less " +
                    "than 'max_timesteps'!"
                )

        if not pType in ["regression", "classification"]:
            raise ValueError("Either 'regression' or 'classification' only!")
        else:
            self.pType = pType

        # Reward function
        self.fQueryCost = fQueryCost
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
            background_dataset=None, **kwargs
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
            the sampled batches, for debugging purposes.

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
        self.sample_weight = sample_weight

        # If user wants to monitor progression of terms in the loss function
        if returnQ:
            Q_avr_list = []
            r_avr_list = []
            V_avr_list = []

        # Enforce float32
        X = X.astype(np.float32)

        # Compute background dataset if needed
        if background_dataset is None:
            # Computing background dataset (assuming numerical features)
            df_train_avg = pd.DataFrame(
                data=np.zeros(X.shape), index=X.index,
                columns = X.columns
            )
            for i in df_train_avg.index:
                df_train_avg.loc[i] = X.drop(i).mean(axis=0)

            df_train_avg.loc["Total", :] = X.mean(axis=0)
        else:
            df_train_avg = background_dataset

        # Initializing the environment
        self.env = Environment(
            X, y, df_train_avg,
            self.fQueryCost, self.mQueryCost,
            self.fRepeatQueryCost, self.p_wNoFCost, self.errorCost,
            self.pType, self.regression_tol, self.regression_error_rounding,
            self.pModels, self.device
        )
        self.env.reset()

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
                len(self.env.state), len(self.env.actions), nLayer1, nLayer2
            ).to(self.device)

            self.target_net = DQN(
                len(self.env.state), len(self.env.actions), nLayer1, nLayer2
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initializing the optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

        # Create dictionary to save information per episode
        doc = defaultdict(dict)

        # Training the agent over self.episodes
        if self.max_timesteps is None:
            self.max_timesteps = self.env.nFeatures * 3

        for i_episode in range(self.episodes):
            print(f"\n\n=== Episode {i_episode+1} === === ===")
            state = self.env.reset()

            # Convert state to pytorch tensor
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            for t in count():
                # Make agent take an action
                action = self.select_action(state, self.env)

                if t > self.max_timesteps:
                    action = torch.tensor([[-1]], device=self.device)

                # Agent carries out action on the environment and returns:
                # - observation (state in next time-step)
                # - reward
                observation, reward, terminated = self.env.step(
                    action.item(), sample_weight=self.sample_weight, **kwargs
                )

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
                _res = self.optimize_model(loss_function, returnQ)
                if returnQ:
                    if not _res is None:
                        Q_avr_list.append(_res[0])
                        r_avr_list.append(_res[1])
                        V_avr_list.append(_res[2])

                # Apply soft update to target network's weights
                targetParameters = self.target_net.state_dict()
                policyParameters = self.policy_net.state_dict()

                for key in policyParameters:
                    targetParameters[key] = policyParameters[key]*self.tau + \
                        targetParameters[key]*(1 - self.tau)

                self.target_net.load_state_dict(targetParameters)

                if terminated:
                    doc_episode = {
                        "SampleID": self.env.X_test.index[0],
                        "y_true": self.env.y_test,
                        "y_pred": self.env.y_pred,
                        "PredModel": self.env.get_prediction_model(),
                        "Episode": i_episode + 1,
                        "Iterations": t+1,
                        "Mask": self.env.get_feature_mask(),
                        "predModel_nChanges": self.env.pm_nChange
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
                    torch.save(
                        self.policy_net.state_dict(),
                        f"agentPolicy_nE{i_episode + 1}.pt"
                    )

        if returnQ:
            Q_avr_list.append(_res[0])
            r_avr_list.append(_res[1])
            V_avr_list.append(_res[2])
            return doc, (Q_avr_list, r_avr_list, V_avr_list)
        else:
            return doc

    def predict(self, X, **kwargs):
        '''
        Use trained agent to select features and a suitable prediction model
        to predict the target/class, given X.

        Parameters
        ----------
        X : pd.DataFrame
            Test samples

        Returns
        -------
        y : array
            Target/Class predicted for X

        doc_test : dict
            Log/documentation of each test sample
        '''
        # Create dictionary to save information per episode
        doc_test = defaultdict(dict)

        # Array to store predictions
        y_pred = np.zeros(X.shape[0])

        for i, test_sample in enumerate(X.index):
            print(f"\n\n=== Test sample {test_sample} === === ===")
            state = self.env.reset(sample=X.loc[[test_sample]])

            # Convert state to pytorch tensor
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            for t in count():
                action = self.select_action(state, self.env)

                if t > self.max_timesteps:
                    action = torch.tensor([[-1]], device=self.device)

                observation, reward, terminated = self.env.step(
                    action.item(), sample_weight=self.sample_weight, **kwargs
                )

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
                        "PredModel": self.env.get_prediction_model(),
                        "Iterations": t+1,
                        "Mask": self.env.get_feature_mask(),
                        "predModel_nChanges": self.env.pm_nChange
                    }
                    doc_test[test_sample] = doc_episode

                    print("Episode terminated:")
                    print(
                        f"- Iterations                 : {doc_episode['Iterations']}\n" +
                        f"- Features selected          : {doc_episode['Mask'].sum()}\n" +
                        f"- Prediction model           : {doc_episode['PredModel']}\n" +
                        f"- Prediction model #(change) : {doc_episode['predModel_nChanges']}"
                    )
                    y_pred[i] = self.env.y_pred
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

    def optimize_model(self, loss_function, returnQ):
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

        # Optimize the model (policy network)
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        if returnQ:
            Q_avr = state_action_values.detach().numpy().mean()
            r_avr = reward_batch.unsqueeze(1).numpy().mean()
            V_avr = expected_state_action_values.unsqueeze(1).numpy().mean()
            res = (Q_avr, r_avr, V_avr)
        else:
            res = None

        return res
