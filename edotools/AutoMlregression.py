import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
#from sklearn.impute import SimpleImputer
#from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

#from sklearn.learning_curve import validation_curve
#from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score ,roc_auc_score


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


from collections.abc import Iterable
from itertools import product
import time


from mytransformers import FeatureSelector, ModifiedSimpleImputer, ModifiedFeatureUnion, MyLEncoder


from .BaseAutoMlEstimator import BaseAutoMlEstimator


class AutoMlregression(BaseAutoMlEstimator):    
    
    def __init__(self, df, X_train, X_test, y_train, y_test, target_col, reports_path='/reports'):
        super().__init__(df, X_train, X_test, y_train, y_test, target_col, reports_path='/reports')
        
