import pandas as pd
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

class load_datasets:
    """
    class that loads the datasets for the study
    """ 
    def __init__(self):
        pass
    
    def regression_dataset(self):
        """
        Makes a random regression data set, given the number of samples and features, 
        and splits them for training and test
        Xr: input samples
        yr: output samples
        """

        # Obtaining the regression data
        Xr, yr = make_regression(n_samples=500, n_features=10, noise=10)
        Xr = pd.DataFrame(Xr)
        yr = pd.DataFrame(yr)
        
        # Split the datasets into train and test 
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr)
        return Xr_train, Xr_test, yr_train, yr_test
    
    def classification_dataset(self):
        """
        Makes a random classification data set, given the number of samples and features, 
        and splits them for training and test
        Xr: input samples
        yr: output samples
        """

        # Obtaining the classification data 
        Xc, yc = make_classification(n_samples=500, n_features=10, n_classes=2, n_redundant=1)
        Xc = pd.DataFrame(Xc)
        yc = pd.DataFrame(yc)

        # Split the datasets into train and test
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc)
        return Xc_train, Xc_test, yc_train, yc_test

class select_features: 
    """
    class for the feature selection 
    """
    def __init__(self):
        pass

    def selection(self, X_train, y_train):
        """
        Selects the features that are relevant: when the Lasso coefficients 
        don't go to zero
        selected_feats: array with the tag of the relevant features
        """
        # Using select from model to obtain the feature importance
        sel_= SelectFromModel(Lasso(alpha=0.005,random_state=0))
        # train Lasso model and select features
        sel_.fit(X_train,y_train)

        # let's print the number of total and selected features
        selected_feats = X_train.columns[(sel_.get_support())]
        # let's print some stats
        #print('Total features: {}'.format((X_train.shape[1])))
        #print('selected features: {}'.format(len(selected_feats)))
        #print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_== 0)))
        return selected_feats

    def df_feature_selection(self,X_train, X_test, sel_features=None, importance=None):
        """
        Keeps only the features that were selected on the samples data frames
        Returns the X for train and test only for the selected features
        """
        # loop that allows to have another threshold for feature importance that is not 
        # just zero
        if sel_features is None:
            sel_features = []
            for idx,val in enumerate(importance):
                if val > 0.01:
                    sel_features.append(idx)

        # select the relevant data
        X_train = X_train[sel_features]
        X_test = X_test[sel_features]
        return X_train, X_test

