from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class RandomForest_Classification:
    def __init__(self):
        self.params_rf = {'n_estimators': 10, 'random_state' : 42}

    def RF_model(self):
        model_rf = RandomForestClassifier(**self.params_rf)
        return model_rf

class GradientBoosting_Classification:
    def __init__(self):
        self.params_gb = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'loss': 'ls'}

    def GB_model(self):
        model_gb = GradientBoostingClassifier()
        return model_gb

class Stacking_Classification(RandomForest_Classification, GradientBoosting_Classification):
    def __init__(self):
        GradientBoosting_Classification.__init__(self)
        RandomForest_Classification.__init__(self)
        
        model_gb = RandomForestClassifier()
        model_rf = GradientBoostingClassifier(**self.params_rf)

        self.estimators = [('Random Forest', model_rf),
                            ('Gradient Boosting', model_gb)]

    def Stack_model(self):
        model_stack = StackingClassifier(estimators=self.estimators, 
                                        final_estimator=LogisticRegression())
        return model_stack