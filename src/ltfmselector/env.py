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
            self, X, y, X_bg, fQueryCost, mQueryCost,
            fRepeatQueryCost, p_wNoFCost, errorCost, pType,
            regression_tol, pModels, device
    ):
        '''
        The environment with which the agent interacts, including the actions
        that the agent may take.

        Paramters
        ---------
        X : pd.DataFrame
         - Training dataset, pandas dataframe with the shape:
           (n_samples, n_features)

        y : pd.Series
         - Class/Target vector

        X_bg : pd.DataFrame
         - Background dataaset, pandas dataframe with the shape:
           (n_samples+1, n_features)

           An extra row for 'Total', average feature values for all training
           samples

        fQueryCost : float
         - Cost of querying a feature

        mQueryCost : float
         - Cost of querying a prediction model

        fRepeatQueryCost : float
         - Cost of querying a feature already previously selected

        p_wNoFCost : float
         - Cost of switching selected prediction model

        errorCost : float
         - Cost of making a wrong prediction

           If pType == 'regression', then
           Agent is punished -errorCost*abs(``prediction`` - ``target``)

           If pType == 'classification', then
           Agent is punished -errorCost

        pType : {'regression' or 'classification'}
         - Type of prediction to make

        regression_tol : float
         - Only applicable for regression models, punish agent if prediction
           error is bigger than regression_tol

        pModels : None or ``list of prediction models``
         - Options of prediction models that the agent can choose from.

        device : ``CPU`` or ``GPU``
         - Computation device
        '''
        # Datasets
        self.X = X
        self.y = y
        self.X_bg = X_bg

        # Reward functions
        self.fQueryCost = fQueryCost
        self.mQueryCost = mQueryCost
        self.fRepeatQueryCost = fRepeatQueryCost
        self.p_wNoFCost = p_wNoFCost
        self.errorCost = errorCost
        self.regression_tol = regression_tol

        self.device = device

        # Available prediction models
        self.pType = pType
        self.pModels = pModels

        # Metadata of training dataset
        self.nSamples = self.X.shape[0]
        self.nFeatures = self.X.shape[1]

        # Agent's actions
        self.actions = np.arange(-1, self.nFeatures + len(self.pModels))

        # Counter for prediction model change
        self.pm_nChange = 0

        self.state = None

    def reset(self, sample=None):
        '''
        Resetting the environment:
        1. State is initialized
        2. Actions are initialized
        3. If training, a random sample is selected from the training dataset
        4. If test, a test sample is passed as sample
        '''
        # Training the agent
        if sample is None:
            # Random sample from X
            i = random.randint(0, self.nSamples - 1)

        # Reset actions list
        self.actions = np.arange(-1, self.nFeatures + len(self.pModels))

        # Reset regressor result
        self.y_pred = None

        if sample is None:
            # Features and target/class of randomly selected patient
            self.X_test = self.X.iloc[[i]]
            self.y_test = (self.y.iloc[[i]]).iloc[0]

            # Background dataset computed based on average feature values of
            # training dataset, with sample i exempted
            self.X_avg = self.X_bg.iloc[[i]]

            # Training dataset, with sample i exempted
            self.X_train = self.X.drop(self.X.index[i])
            self.y_train = self.y.drop(self.X.index[i])

        else:
            # Test sample passed by user
            self.X_test = sample
            self.y_test = None # indicating this is a test sample to predict

            # Background dataset computed based on average feature values of
            # training dataset
            self.X_avg = self.X_bg.loc[["Total"]]

            # Training dataset, using original X and y
            self.X_train = self.X
            self.y_train = self.y

        # Formulating the state (partially observable MDP)
        self.state = np.concatenate(
            (
                self.X_avg.to_numpy().reshape(-1),
                np.zeros(self.nFeatures),
                [random.randint(0, len(self.pModels)-1)]
            )
        )

        # Counter for prediction model change
        self.pm_nChange = 0

        return self.state

    def step(self, action, sample_weight=None, **kwargs):
        '''
        Agent carries out an action.

        Parameters
        ----------
        action : int
         = -1 (make a prediction with selected features and prediction model)
         = int : [0, n_features] (query a feature)
         = int : [n_features, n_features + n_model] (query a prediction model)

        sample_weight : list or array or None
         - Per-sample weights
        '''
        # === === === ===
        # Query a feature
        if ((action > -1) & (action < self.nFeatures)):
            # If feature has NOT been selected, select that feature
            if self.state[self.nFeatures + action] == 0:

                # 1. Set mask value to 1
                self.state[self.nFeatures + action] = 1

                # 2. Set feature value to that of the patient's
                self.state[action] = self.X_test.iloc[0, action]

                # Punish for querying a feature
                return [self.state, -self.fQueryCost, False]

            # Punish agent for attempting to query a feature already
            # previously selected
            else:
                return [self.state, -self.fRepeatQueryCost, False]

        # === === === ===
        # Changing prediction model in the state
        elif (action >= self.nFeatures):
            # This returns the index/option of prediction model
            self.state[-1] = action - self.nFeatures

            self.pm_nChange += 1

            return [self.state, -self.mQueryCost, False]

        else:
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
                return [None, -self.p_wNoFCost, True]

            # === === === ===
            # Make a prediction with selected features and prediction model

            ### Special-tailored implementation ###
            if "smsproject" in list(kwargs.keys()):
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

            # Get selected prediction model
            selected_predModel = self.pModels[int(self.state[-1])]

            ### Special-tailored implementation ###
            if "smsproject" in list(kwargs.keys()):
                X_train_wLabel = X_train.copy()
                X_train_wLabel["Target"] = self.y.loc[X_train_wLabel.index]

                sample_weight = balance_classDistribution_patient(
                    X_train_wLabel, "Target"
                ).to_numpy(dtype=np.float32)[:,0]

            # Convert X_train and y_train into numpy arrays if they are Pandas
            # DataFrame or Series
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values

            if isinstance(y_train, pd.Series):
                y_train = y_train.values

            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values

            if sample_weight is None:
                selected_predModel.fit(X_train, y_train)
            else:
                selected_predModel.fit(
                    X_train, y_train, sample_weight=sample_weight
                )
                
            self.y_pred = selected_predModel.predict(X_test)[0]

            if "smsproject" in list(kwargs.keys()):
                # Capping values between 0 and 3
                self.y_pred = capUpperValues(self.y_pred)
                self.y_pred = capLowerValues(self.y_pred)

            # Training
            if not self.y_test is None:
                if self.pType == "regression":
                    if abs(self.y_pred - self.y_test) < self.regression_tol:
                        print("\nCorrect prediction")
                        penalty = 0
                    else:
                        print("\nIncorrect prediction")
                        penalty = -self.errorCost * abs(self.y_pred - self.y_test)

                elif self.pType == "classification":
                    if self.y_pred == self.y_test:
                        print("\nCorrect prediction")
                        penalty = 0
                    else:
                        print("\nIncorrect prediction")
                        penalty = -self.errorCost

                print(f"True Output: {self.y_test} | Prediction: {self.y_pred}\n")

                return [None, penalty, True]

            # Test
            else:
                print(f"Prediction: {self.y_pred}\n")

                return [None, 0.0, True]

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

    def get_prediction_model(self):
        '''
        Get the selected prediction model and returns its index
        '''
        return int(self.state[-1])
