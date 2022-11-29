from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class plots_classification:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_confussion(self, models):
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        axs = np.ravel(axs)

        for ax, (name, est) in zip(axs, models):
            
            est.fit(self.X_train, self.y_train)
            plot_confusion_matrix(est,self.X_test, self.y_test, ax=ax,  normalize='true')
            ax.title.set_text(name)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
        plt.savefig("plots/confussion_classification.png") 
        plt.show()

    def plot_feature_importance(self, selected_feats):
        lin_model = Lasso(alpha=0.005, random_state=0)
        model = lin_model.fit(self.X_train, self.y_train)
        importance = pd.Series(np.abs(lin_model.coef_.ravel()))
        importance.index = selected_feats
        importance.sort_values(inplace=True, ascending=False)
        importance.plot.bar(figsize=(8,5))
        plt.ylabel('Lasso Coefficients')
        plt.title('Feature Importance')
        plt.savefig("plots/features_classification.png")
        plt.show()
        return importance