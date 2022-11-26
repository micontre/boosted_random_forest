import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


class load_datasets: 
    def __init__(self):
        pass
    
    def regression_dataset(self):
        Xr, yr = datasets.load_diabetes(return_X_y=True)
        columns_reg = ["age", "sex", "bmi", "blood_pressure", "s1_tc", 
                        "s2_ldl", "s3_hdl", "s4_tch", "s5_ltg", "s6_glu", "diabetes_progression"]
        Xr = pd.DataFrame(Xr, columns=columns_reg[0:10])
        yr = pd.DataFrame(yr, columns=columns_reg[10:])
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr)
        return Xr_train, Xr_test, yr_train, yr_test
    
    def classification_dataset(self):
        Xc, yc = datasets.load_breast_cancer(return_X_y=True)
        columns_class = ["radius", "texture", "perimeter", "area", "smooth", "compact", "concavity",
                        "concave_pts", "symmetry", "fractal_dim", "radius_std", "texture_std", 
                        "perimeter_std", "area_std", "smooth_std", "compact_std", "concavity_std", 
                        "concave_pts_std", "symmetry_std", "fractal_dim_std", "radius_max", 
                        "texture_max", "perimeter_max", "area_max", "smooth_max", "compact_max", 
                        "concavity_max", "concave_pts_max", "symmetry_max", "fractal_dim_max", 
                        "benign_tumor"]
        Xc = pd.DataFrame(Xc, columns=columns_class[0:30])
        yc = pd.DataFrame(yc, columns=columns_class[30:])
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, stratify=yc, random_state=42)
        return Xc_train, Xc_test, yc_train, yc_test

class select_features: 
    def __init__(self):
        pass

    def selection(self, X_train, y_train):
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

    def df_feature_selection(self,X_train, X_test, sel_feats):
        X_train = X_train[sel_feats]
        X_test = X_test[sel_feats]
        return X_train, X_test

