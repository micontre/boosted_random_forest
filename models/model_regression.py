from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

class RandomForest_Regression:
    def __init__(self):
        self.params_rf = {'n_estimators': 400, 'random_state' : 0}

    def RF_model(self):
        model_rf = RandomForestRegressor(**self.params_rf)
        return model_rf

class GradientBoosting_Regression:
    def __init__(self):
        self.params_gb = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

    def GB_model(self):
        model_gb = GradientBoostingRegressor(**self.params_gb)
        return model_gb

class Stacking_Regression(RandomForest_Regression, GradientBoosting_Regression):
    def __init__(self):
        GradientBoosting_Regression.__init__(self)
        RandomForest_Regression.__init__(self)
        
        model_gb = GradientBoostingRegressor(**self.params_gb)
        model_rf = RandomForestRegressor(**self.params_rf)

        self.estimators = [('Random Forest', model_rf),
                            ('Gradient Boosting', model_gb)]

    def Stack_model(self):
        model_stack = StackingRegressor(estimators=self.estimators)
        return model_stack

