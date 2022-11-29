import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
import warnings

class RandomForest_Regression:
    def __init__(self):
        self.params_rf = {
                            'n_estimators': 1000,
                            'max_depth': 10}

    def RF_model(self):
        model_rf = RandomForestRegressor(**self.params_rf)
        return model_rf
    
    def RF_hyperparameters(self, Xr_train, yr_train):
        RF = RandomForestRegressor()
        parameters = {
                        'n_estimators' : [100,500,1000, 1500],
                        'max_depth'    : [4,6,8,10]
                        }
        grid_RF = GridSearchCV(estimator=RF, param_grid = parameters, cv = 2, n_jobs=-1)
        grid_RF.fit(Xr_train, yr_train)
        print("==============================================================")
        print(" Results from Grid Search for Random Forest" )
        print("\n The best estimator across ALL searched params:\n",grid_RF.best_estimator_)
        print("\n The best score across ALL searched params:\n",grid_RF.best_score_)
        print("\n The best parameters across ALL searched params:\n",grid_RF.best_params_)
        warnings.filterwarnings(action='ignore')
        return grid_RF.best_estimator_

class GradientBoosting_Regression:
    def __init__(self):
        self.params_gb = {
                            'learning_rate': 0.01, 
                            'max_depth': 4, 
                            'n_estimators': 1500, 
                            'subsample': 0.1}

    def GB_model(self):
        model_gb = GradientBoostingRegressor(**self.params_gb)
        return model_gb
    
    def GB_hyperparameters(self, Xr_train, yr_train):
        GBR = GradientBoostingRegressor()
        parameters = {
                        'learning_rate': [0.01,0.02,0.03,0.04],
                        'subsample'    : [0.9, 0.5, 0.2, 0.1],
                        'n_estimators' : [100,500,1000, 1500],
                        'max_depth'    : [4,6,8,10]
                        }
        grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 2, n_jobs=-1)
        grid_GBR.fit(Xr_train, yr_train)
        print("==============================================================")
        print(" Results from Grid Search for Gradient Boosting" )
        print("\n The best estimator across ALL searched params:\n",grid_GBR.best_estimator_)
        print("\n The best score across ALL searched params:\n",grid_GBR.best_score_)
        print("\n The best parameters across ALL searched params:\n",grid_GBR.best_params_)
        warnings.filterwarnings(action='ignore')
        return grid_GBR.best_estimator_

class Stacking_Regression(RandomForest_Regression, GradientBoosting_Regression):
    def __init__(self):
        GradientBoosting_Regression.__init__(self)
        RandomForest_Regression.__init__(self)
        
        model_gb = GradientBoostingRegressor(**self.params_gb)
        model_rf = RandomForestRegressor(**self.params_rf)

        self.estimators = [('Random Forest', model_rf),
                            ('Gradient Boosting', model_gb)]

    def Stack_model(self, estimators=None):
        if estimators is None:
            model_stack = StackingRegressor(estimators=self.estimators, 
                                            final_estimator=RidgeCV())
        else:
            model_stack = StackingRegressor(estimators=estimators, 
                                            final_estimator=RidgeCV())
        return model_stack