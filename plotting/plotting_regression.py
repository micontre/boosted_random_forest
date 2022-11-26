import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Lasso


class plots_regression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_comparison_regression(self, models):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        axs = np.ravel(axs)
        errors_list=[]

        for ax, (name, est) in zip(axs, models):
            start_time = time.time()
            
            model = est.fit(self.X_train, self.y_train)
                            
            elapsed_time = time.time() - start_time
            
            pred = model.predict(self.X_test)
            y_test_flat = [item for sublist in self.y_test.values for item in sublist]
            errors = y_test_flat - pred
            errors_list.append(errors)
            self.errors_list = errors_list
            test_r2 = r2_score(np.exp(self.y_test), np.exp(pred))
            test_rmsle=math.sqrt(mean_squared_log_error(self.y_test,pred))
            self.plot_regression_results(ax,self.y_test,pred,name,(r'$R^2={:.3f}$' + '\n' +
                                    r'$RMSLE={:.3f}$').format(test_r2,test_rmsle),elapsed_time)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
        plt.savefig("plots/vs_regression.png") 
        plt.show()

    def plot_regression_results(self, ax, y_true, y_pred, title, scores, elapsed_time):
        """Scatter plot of the predicted vs true targets."""
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
        titles = ['Random Forest', 'Gradient Boosting','Stacking Regressor']
        f,a = plt.subplots(1,3, figsize=(18, 6))
        a = a.ravel()
        for idx,ax in enumerate(a):
            ax.hist(self.errors_list[idx])
            ax.set_title(titles[idx])
        plt.tight_layout()
        plt.savefig("plots/error_regression.png")
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
        plt.savefig("plots/features_regression.png")
        plt.show()