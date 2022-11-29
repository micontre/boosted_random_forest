from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings

class RandomForest_Classification:
    def __init__(self):
        self.params_rf = {'n_estimators': 10, 'random_state' : 42}

    def RF_model(self):
        model_rf = RandomForestClassifier(**self.params_rf)
        return model_rf
    
    def RF_hyperparameters(self, Xc_train, yc_train):
        RF = RandomForestClassifier()
        parameters = {
                        'n_estimators' : [100,500,1000,1500],
                        'max_depth'    : [4,6,8,10]
                        }
        grid_RF = GridSearchCV(estimator=RF, param_grid = parameters, cv = 2, n_jobs=-1)
        grid_RF.fit(Xc_train, yc_train)
        print("==============================================================")
        print(" Results from Grid Search for Random Forest" )
        print("\n The best estimator across ALL searched params:\n",grid_RF.best_estimator_)
        print("\n The best score across ALL searched params:\n",grid_RF.best_score_)
        print("\n The best parameters across ALL searched params:\n",grid_RF.best_params_)
        warnings.filterwarnings(action='ignore')
        return grid_RF.best_estimator_

class GradientBoosting_Classification:
    def __init__(self):
        self.params_gb = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'loss': 'ls'}

    def GB_model(self):
        model_gb = GradientBoostingClassifier()
        return model_gb

    def GB_hyperparameters(self, Xc_train, yc_train):
        GBC = GradientBoostingClassifier()
        parameters = {
                        'learning_rate': [0.01,0.02,0.03,0.04],
                        'subsample'    : [0.9, 0.5, 0.2, 0.1],
                        'n_estimators' : [100,500,1000, 1500],
                        'max_depth'    : [4,6,8,10]
                        }
        grid_GBC = GridSearchCV(estimator=GBC, param_grid = parameters, cv = 2, n_jobs=-1)
        grid_GBC.fit(Xc_train, yc_train)
        print("==============================================================")
        print(" Results from Grid Search for Gradient Boosting" )
        print("\n The best estimator across ALL searched params:\n",grid_GBC.best_estimator_)
        print("\n The best score across ALL searched params:\n",grid_GBC.best_score_)
        print("\n The best parameters across ALL searched params:\n",grid_GBC.best_params_)
        warnings.filterwarnings(action='ignore')
        return grid_GBC.best_estimator_

class Stacking_Classification(RandomForest_Classification, GradientBoosting_Classification):
    def __init__(self):
        GradientBoosting_Classification.__init__(self)
        RandomForest_Classification.__init__(self)
        
        model_gb = RandomForestClassifier()
        model_rf = GradientBoostingClassifier(**self.params_rf)

        self.estimators = [('Random Forest', model_rf),
                            ('Gradient Boosting', model_gb)]

    def Stack_model(self, estimators=None):
        if estimators is None:
            model_stack = StackingClassifier(estimators=self.estimators, 
                                             final_estimator=LogisticRegression())
        else: 
            model_stack = StackingClassifier(estimators=estimators, 
                                             final_estimator=LogisticRegression())
        return model_stack