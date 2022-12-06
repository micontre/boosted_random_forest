import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso


class plots_regression:
    """
    plotter of the regression figures
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_comparison_regression(self, models):
        """
        Plotter for subplotting on a single figure the comparison of true vs predicted
        values for regression
        """

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        
        axs = np.ravel(axs)
        errors_list=[]

        for ax, (name, est) in zip(axs, models):
            start_time = time.time()
            
            # fitting the model
            model = est.fit(self.X_train, self.y_train)

            # obtaining the time                 
            elapsed_time = time.time() - start_time
            
            # obtaining the predictions, and errors
            pred = model.predict(self.X_test)
            y_test_flat = [item for sublist in self.y_test.values for item in sublist]
            errors = y_test_flat - pred
            errors_list.append(errors)
            self.errors_list = errors_list
            y_scaled = minmax_scale(self.y_test, feature_range=(0,1))
            pred_scaled = minmax_scale(pred, feature_range=(0,1))
            test_r2 = r2_score(np.exp(y_scaled), np.exp(pred_scaled))
            test_rmsle=math.sqrt(mean_squared_log_error(y_scaled,pred_scaled))

            # plotting
            self.plot_regression_results(ax,self.y_test,pred,name,(r'$R^2={:.3f}$' + '\n' +
                                    r'$RMSLE={:.3f}$').format(test_r2,test_rmsle),elapsed_time)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
        plt.savefig("plots/vs_regression.png") 
        plt.show()

    def plot_regression_results(self, ax, y_true, y_pred, title, scores, elapsed_time):
        """
        Scatter plot of the predicted vs true targets
        """
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                '--r', linewidth=2)
        ax.scatter(y_true, y_pred, alpha=0.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlabel('True value')
        ax.set_ylabel('Predicted value')
        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
        ax.legend([extra], [scores], loc='upper left')
        title = title + ' Evaluation in {:.2f} seconds'.format(elapsed_time)
        ax.set_title(title)
    
    def plot_error(self,models):
        """
        Plotter for the error between predicted value and true value
        """
        titles = ['Random Forest Regressor', 'Gradient Boosting Regressor','Stacking Regressor (RF+GB)', 'Stacking Regressor (GB+RF)' ]
        f,a = plt.subplots(2,2, figsize=(12, 9))
        a = a.ravel()
        for idx,ax in enumerate(a):
            # plotting the histograms of errors 
            ax.hist(self.errors_list[idx])
            ax.set_title(titles[idx])
            ax.set(xlabel="Error: y - f(x)")
        plt.tight_layout()
        plt.savefig("plots/error_regression.png")
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
        plt.savefig("plots/features_regression.png")
        plt.show()
        return importance