import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 



class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X[self.columns])
    

    
class ModifiedSimpleImputer(SimpleImputer):
    
#     def fit_transform(self, X,y=None,  **fit_params):
#         self.fit(X,y,  **fit_params)
#         return self.transform(X)

    
    def transform(self, X):
        return pd.DataFrame(super().transform(X))
    

class ModifiedFeatureUnion(FeatureUnion):
    
#     def fit_transform(self, X,y=None,  **fit_params):
#         self.fit(X,y,  **fit_params)
#         return self.transform(X)
    def merge_dataframes_by_column(self, X):
        return pd.concat(X, axis="columns", copy=False)
    
    def transform(self, X):
        #X = self.merge_dataframes_by_column(X)
        return pd.DataFrame(super().transform(X))
    
    
class MyLEncoder():
    
    def transform(self, X, **fit_params):
        enc = LabelEncoder()
        enc_data = []
        for i in list(X.columns):
            X[i] = X[i].astype(str)
            encc = enc.fit(X[i])
            enc_data.append(encc.transform(X[i]))
        return np.asarray(enc_data).T
    
    def fit_transform(self, X,y=None,  **fit_params):
        self.fit(X,y,  **fit_params)
        return self.transform(X)
    def fit(self, X, y, **fit_params):
        return self 
    
    


 