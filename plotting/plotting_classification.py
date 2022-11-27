from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


class plots_classification:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_confussion(self, models):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        #axs = np.ravel(axs)

        for ax, (name, est) in zip(axs, models):
            
            est.fit(self.X_train, self.y_train)
            #proba = model.predict_proba(self.X_test)
            #pred = np.where(proba[:,1] > 0.5, 1, 0)        
            #pred = model.predict(self.X_test)
            plot_confusion_matrix(est,self.X_test, self.y_test, ax=ax,  normalize='true')
            ax.title.set_text(name)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
        plt.savefig("plots/confussion_classification.png") 
        plt.show()