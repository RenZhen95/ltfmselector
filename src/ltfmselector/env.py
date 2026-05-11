import sys
import random
import numpy as np
import pandas as pd

from .utils import balance_classDistribution_patient

### Special-tailored implementation ###
# Function to get patient's ID given a stride pair ID
getPatientID = lambda x: x[0:5] if x.startswith("ES") else x[0:8]

# Functions to clip predicted regression values
capUpperValues = lambda x: 3.0 if x > 3.0 else x
capLowerValues = lambda x: 0.0 if x < 0.0 else x

class Environment:
    def __init__(
            self, X, y, X_bg, y_pred_bg,
            fQueryCost, fQueryFunction,
            fThreshold, fCap, fRate,
            fRepeatQueryCost, p_wNoFCost, errorCost, pType, iniFP,
            regression_tol, regression_error_rounding, pModels, device,
            sample_weight=None, **kwargs
    ):
        '''
        The environment with which the agent interacts, including the actions
        that the agent may take.

        Paramters
        ---------
        X : pd.DataFrame
            Training dataset, pandas dataframe with the shape:
            (n_samples, n_features)

        y : pd.Series
            Class/Target vector

        X_bg : pd.DataFrame
            Background dataset, pandas dataframe with the shape:
            (n_samples+1, n_features)

            An extra row for 'Total', average feature values for all training
            samples

        y_pred_bg : ``float``
           Background prediction if agent predicts without recruiting any
           feature

        fQueryCost : float
            Cost of querying a feature

        fQueryFunction : None or {'step', 'linear', 'quadratic', 'exponential'}
            Function to progressively increase cost of recruiting a feature

        fThreshold : None or int
            If `fQueryFunction == {'step', 'linear', 'quadratic', 'exponential'}`
            Threshold of number of features, before cost of recruiting
            increases

        fCap : None or float
            If `fQueryFunction == {'step'}`, upper limit of penalty

        fRate : None or float
            If `fQueryFunction == {'linear', 'quadratic', 'exponential'}`, rate
            individual cost functions

        fRepeatQueryCost : float
            Cost of querying a feature already previously selected

        p_wNoFCost : float
            Cost of making a prediction without any recruited features

        errorCost : float
            Cost of making a wrong prediction

            If pType == 'regression', then
            Agent is punished -errorCost*abs(``prediction`` - ``target``)

            If pType == 'classification', then
            Agent is punished -errorCost

        pType : {'regression' or 'classification'}
            Type of prediction to make

        iniFP : numpy.ndarray
            Probability of each feature being selected as an initial feature

        regression_tol : float
            Only applicable for regression models, punish agent if prediction
            error is bigger than regression_tol

        regression_error_rounding : int
            Only applicable for regression models. The error between the
            prediction and true value is rounded to the input decimal place.

        pModels : None or ``list of prediction models``
            Options of prediction models that the agent can choose from.

        device : ``CPU`` or ``GPU``
            Computation device

        sample_weight : list or array or None
            Per-sample weights
        '''
        # Datasets
        self.X = X
        self.y = y
        self.X_bg = X_bg
        self.y_pred_bg = y_pred_bg

        # Reward functions
        self.fQueryCost = fQueryCost
        self.fQueryFunction = fQueryFunction
        self.fThreshold = fThreshold
        self.fCap = fCap
        self.fRate = fRate

        self.fRepeatQueryCost = fRepeatQueryCost
        self.p_wNoFCost = p_wNoFCost
        self.errorCost = errorCost
        self.regression_tol = regression_tol
        self.regression_error_rounding = regression_error_rounding

        self.device = device
        self.sample_weight = sample_weight

        # Available prediction models
        self.pType = pType
        self.pModels = pModels

        # Metadata of training dataset
        self.nSamples = self.X.shape[0]
        self.nFeatures = self.X.shape[1]

        # Probability of features being selected as initial feature
        self.p_InitialF = iniFP

        # Agent's actions
        self.actions = np.arange(self.nFeatures + len(self.pModels))

        # Function to check if passed action index is valid
        self.isActionValid = lambda x: True if 0 <= x < len(self.actions) else False

        ### Special-tailored implementation ###
        if "smsproject" in list(kwargs.keys()):
            self.smsproject = True
        else:
            self.smsproject = False

        self.state = None

    def reset(self, sample=None):
        '''
        Resetting the environment:
        1. State is initialized
        2. Actions are initialized
        3. If training, a random sample is selected from the training dataset
        4. If test, a test sample is passed as sample
        '''
        # Get a random sample from the training dataset
        if sample is None:
            # Random sample from X
            i = random.randint(0, self.nSamples - 1)

        # Reset regressor result
        self.y_pred = None

        # Background dataset computed based on average feature values of
        # training dataset
        self.X_avg = self.X_bg.loc[["Total"]]

        if sample is None:
            # Features and target/class of randomly selected patient
            self.X_test = self.X.iloc[[i]]
            self.y_test = (self.y.iloc[[i]]).iloc[0]

            # Training dataset, with sample i exempted to prevent PMs from overfitting
            self.X_train = self.X.drop(self.X.index[i])
            self.y_train = self.y.drop(self.X.index[i])

        else:
            # Test sample passed by user
            self.X_test = sample
            self.y_test = None # indicating this is a test sample to predict

            # Training dataset
            self.X_train = self.X
            self.y_train = self.y

        # Formulating the state (partially observable MDP)
        self.state = np.concatenate(
            (
                self.X_avg.to_numpy().reshape(-1), np.zeros(self.nFeatures)
            )
        )

        # Sample initial feature
        inif = self.sample_initialFeature()

        # Update the state
        self.state[inif] = self.X_test.iloc[0, inif]
        self.state[self.nFeatures + inif] = 1

        return self.state

    def step(self, action):
        '''
        Agent carries out an action.

        Parameters
        ----------
        action : int
            = int : [0, n_features]
                    (query a feature)
            = int : [n_features, n_features + n_model]
                    (use PM to make prediction)
        '''
        # Sanity check
        if not self.isActionValid(action):
            raise TypeError(
                f"Action {action} is not valid. Actions must lie between 0 " +
                f"and {len(self.actions)-1}"
            )

        # QUERY FEATURE
        if action < self.nFeatures:

            # If feature has not yet been selected
            if self.state[self.nFeatures + action] == 0:

                # 1. Set mask value to 1
                self.state[self.nFeatures + action] = 1

                # 2. Set feature value to that of the patient's
                self.state[action] = self.X_test.iloc[0, action]

                # Punish for querying a feature
                return [self.state, -self.get_fQueryCost(), False]

            # Punish agent for attempting to query a feature already
            # previously selected
            else:
                return [self.state, -self.fRepeatQueryCost, False]

        # MAKE PREDICTION
        else:

            # Get selected prediction model
            self.PM = action - self.nFeatures
            selected_predModel = self.pModels[self.PM]

            # Retain only features selected by agent
            X_train = self.X_train.copy()
            y_train = self.y_train.copy()

            X_test = self.X_test.copy()

            # Get feature mask
            mask = self.get_feature_mask()

            col_to_retain = [
                col for i, col in enumerate(X_train.columns) if mask[i]==1
            ]

            # === === === ===
            # Punish agent if it decides to predict without selecting any
            # features
            if len(col_to_retain) == 0:
                self.y_pred = self.y_pred_bg
                return [None, -self.p_wNoFCost, True]

            # === === === ===
            # Make a prediction with selected features and prediction model

            ### Special-tailored implementation ###
            if self.smsproject:
                testpatientID = getPatientID(X_test.index[0])
                otherSP_of_testPatient = [
                    sp for sp in X_train.index if getPatientID(sp) == testpatientID
                ]
                print(
                    f"\n\nTest patient: {X_test.index[0]}\n" +
                    f"- Other stride pairs: {otherSP_of_testPatient}"
                )
                X_train = X_train.drop(otherSP_of_testPatient)
                y_train = y_train.drop(otherSP_of_testPatient)

            X_train = X_train[col_to_retain]
            X_test  = X_test[col_to_retain]

            ### Special-tailored implementation ###
            if self.smsproject:
                X_train_wLabel = X_train.copy()
                X_train_wLabel["Target"] = self.y.loc[X_train_wLabel.index]

                _weights = balance_classDistribution_patient(
                    X_train_wLabel, "Target"
                ).to_numpy(dtype=np.float32)[:,0]
            else:
                _weights = self.sample_weight

            # Convert X_train and y_train into numpy arrays if they are Pandas
            # DataFrame or Series
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values

            if isinstance(y_train, pd.Series):
                y_train = y_train.values

            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values

            # Fit PM
            selected_predModel.fit(X_train, y_train, sample_weight=_weights)

            # Use PM to make a prediction
            self.y_pred = selected_predModel.predict(X_test)[0]

            if self.smsproject:
                # Capping values between 0 and 3
                self.y_pred = capUpperValues(self.y_pred)
                self.y_pred = capLowerValues(self.y_pred)

            # Training
            if not self.y_test is None:
                if self.pType == "regression":
                    if round(
                            abs(self.y_pred - self.y_test),
                            self.regression_error_rounding
                    ) < self.regression_tol:
                        penalty = 0
                    else:
                        penalty = -self.errorCost * abs(self.y_pred - self.y_test)

                elif self.pType == "classification":
                    if self.y_pred == self.y_test:
                        penalty = 0
                    else:
                        penalty = -self.errorCost

                return [None, penalty, True]

            # Test
            else:
                return [None, 0.0, True]

    def get_fQueryCost(self):
        '''
        Get cost of querying a feature
        '''
        if self.fQueryFunction is None:
            return self.fQueryCost

        # Get number of total recruited features
        nFSubset = (self.get_feature_mask()).sum()

        # DEV:: If more than 10 statements, implement dictionary instead
        if self.fQueryFunction == "step":
            return self.get_fQueryCostStep(nFSubset)
        elif self.fQueryFunction == "linear":
            return self.get_fQueryCostLinear(nFSubset)
        elif self.fQueryFunction == "quadratic":
            return self.get_fQueryCostQuadratic(nFSubset)

    def get_fQueryCostStep(self, _nFSubset):
        '''Step function for querying feature'''
        if _nFSubset > self.fThreshold:
            return self.fCap
        else:
            return self.fQueryCost

    def get_fQueryCostLinear(self, _nFSubset):
        '''Linear function for querying feature'''
        _qC = max(
            self.fQueryCost,
            self.fQueryCost + self.fRate*(_nFSubset-self.fThreshold)
        )
        if not self.fCap is None:
            return min(self.fCap, _qC)
        else:
            return _qC

    def get_fQueryCostQuadratic(self, _nFSubset):
        '''Quadratic function for querying feature'''
        if _nFSubset > self.fThreshold:
            _qC = self.fQueryCost + self.fRate*(_nFSubset-self.fThreshold)**2
        else:
            _qC = self.fQueryCost

        if not self.fCap is None:
            return min(self.fCap, _qC)
        else:
            return _qC

    def get_feature_mask(self):
        '''
        Get the (boolean) feature mask that indicates if a feature has
        been selected
        '''
        return self.state[self.nFeatures:self.nFeatures*2]

    def get_random_action(self):
        '''
        Select a random action
        '''
        return random.choice(self.actions)

    def __getstate__(self):
        state = self.__dict__.copy()
        print(state.keys())

        del state['pModels']

    def sample_initialFeature(self):
        '''
        Takes an array, where each element pertains to the probability of sampling
        a particular feature (i.e. p[3] : probability of sampling the 4th feature),
        and returns a sampled feature.

        Returns
        -------
        sampled_idx : int
         - Index of feature
        '''
        # Sample a feature index based on the distribution
        return np.random.choice(
            np.arange(len(self.p_InitialF)), p=self.p_InitialF
        )
