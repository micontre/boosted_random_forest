from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 


class plots_classification:
    """
    plotter of the classification figures
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_confussion(self, models):
        """
        Plotter for subplotting on a single figure the confusion matrices 
        """

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        axs = np.ravel(axs)

        for ax, (name, est) in zip(axs, models):
            start_time = time.time()
            
            # fitting the model
            est.fit(self.X_train, self.y_train)
                            
            # obtaining the time
            elapsed_time = time.time() - start_time
        
            # plotting confusion matrix
            plot_confusion_matrix(est,self.X_test, self.y_test, ax=ax,  normalize='true')
            title = name + ' Evaluation in {:.2f} seconds'.format(elapsed_time)
            ax.title.set_text(title)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
        plt.savefig("plots/confussion_classification.png") 
        plt.show()

    def plot_feature_importance(self, selected_feats):
        """
        Plotter for feature importance
        """
        # building the Lasso model (an unbiased model different than RF and GB)
        lin_model = Lasso(alpha=0.005, random_state=0)
        model = lin_model.fit(self.X_train, self.y_train)

        # obtaining feature importance
        importance = pd.Series(np.abs(lin_model.coef_.ravel()))
        importance.index = selected_feats
        importance.sort_values(inplace=True, ascending=False)

        # plotting the feature importance
        importance.plot.bar(figsize=(8,5))
        plt.ylabel('Lasso Coefficients')
        plt.title('Feature Importance')
        plt.savefig("plots/features_classification.png")
        plt.show()
        return importance