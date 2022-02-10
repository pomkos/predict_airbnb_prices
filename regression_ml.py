import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

class CompareRegressionModels:
    all_models = {
        'linreg': LinearRegression(),
        'lasso': Lasso(),
        'svr': SVR(),
        'dt': DecisionTreeRegressor(),
        'random_forest': RandomForestRegressor()
        }

    def __init__(self, dataframe: pd.DataFrame, feature_columns: List[str], target_column: str):
        '''
        This class will split, train, and test the given dataset and let
        us know how each model performed. Results are accessed via class variables:
        
        self.results: all mse/rmse/mae mean scores
        self.mse: mean mse scores for each model
        self.rmse: mean rmse scores for each model
        self.mae: mean mae scores for each model
        
        Access the trained models using self.fitted_models
        '''
        assert dataframe[feature_columns].isna().sum().sum() == 0, "NaNs detected in feature columns" # models cannot tolerate null values
        assert dataframe[target_column].isna().sum() == 0, 'NaNs detected in target column'

        # initiate score storeage in separate dicts for easier conceptualization
        all_scores = {
            'MSE': {},
            'RMSE': {},
            'MAE': {}
        }

        for model_name in self.all_models.keys():
            all_scores['MSE'][model_name] = []
            all_scores['RMSE'][model_name] = []
            all_scores['MAE'][model_name] = []

        self.all_scores = all_scores # needed by other class functions

        kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
        for idx, t in kf5.split(dataframe):
            df_split = dataframe.iloc[idx,:]

            X = df_split[feature_columns]
            y = df_split[target_column]

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)
            self.y_test, self.y_train, self.X_test, self.X_train = y_test, y_train, X_test, X_train
            self.fit_predict_models(X_train, y_train, X_test, y_test)

        self.calculate_mean_outcomes()

    def fit_predict_models(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
        '''
        Runs and stores the results in a class dictionary
        '''
        all_models = self.all_models
        self.fitted_models = {}
        model_predictions = {}

        for model_name, model in all_models.items():
            fitted_model = model # to initiate model
            fitted_model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)
            model_predictions[model_name] = y_pred
            self.fitted_models[model_name] = fitted_model
            mse, rmse, mae = self.calculate_outcome_measures(y_test, y_pred)
            for score_name, score in zip(['MSE', 'RMSE', 'MAE'], [mse, rmse, mae]):
                self.all_scores[score_name][model_name].append(score)

        self.model_predictions = model_predictions

    def calculate_outcome_measures(self, y_test: np.array, y_pred: np.array) -> Tuple[float]:
        '''
        Calculates the MSE, RMSE, MAE
        '''
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        return mse, rmse, mae

    def calculate_mean_outcomes(self):
        '''
        Calculates the mean MSE, RMSE, MAE for each model's ksplit output
        '''
        results = pd.DataFrame(columns=['score_name', 'model', 'mean_value'])
        for score_name in self.all_scores.keys():
            for model_name in self.all_scores[score_name].keys():
                kscores = self.all_scores[score_name][model_name]
                mean_kscore = sum(kscores) / len(kscores)

                results = results.append({
                    'model': model_name,
                    'score_name': score_name,
                    'mean_value': mean_kscore
                }, ignore_index=True)
        self.results = results
        
        # for easier accessibility by user
        self.mse = results[results['score_name']=='MSE'].reset_index(drop=True)
        self.rmse = results[results['score_name']=='RMSE'].reset_index(drop=True)
        self.mae = results[results['score_name']=='MAE'].reset_index(drop=True)