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


class BaseAutoMlEstimator:
    """Base class for all estimators in scikit-learn.
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """
    
    def __init__(self, df, X_train, X_test, y_train, y_test, target_col, reports_path='/reports'):
        self.df = df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_col = target_col        
        self.reports_path = reports_path

        
    @classmethod
    def get_params_combination(self, param_grid):
        iterator = product(*[v if isinstance(v, Iterable) else [v] for v in param_grid.values()])
        return [dict(zip(param_grid.keys(), values)) for values in iterator]
    
    
    def get_report_about_features(self,df, reports_path='/reports'):
        
        ### Формируем отчет по фичам DF
        dict_null = df.isnull().sum().to_dict()
        dict_type = df.dtypes.to_dict()

        dict_unique = dict()
        for column in df.columns:
            dict_unique[column] = len(df[column].unique())

        report_df = df.describe(include='all').T.rename_axis('features').reset_index()
        report_df['feat_count_null'] = report_df['features'].map(dict_null)
        report_df['feat_type'] = report_df['features'].map(dict_type)
        report_df['unique'] = report_df['features'].map(dict_unique)
        
        report_df.to_csv('..'+ reports_path + '/features_report.csv', index=False)   
        
        print("Отчет по статистикам по фичам сформирован")
        return report_df

    
    
    def optimize_types(self, df, inplace=False):
    
        np_types = [np.int8 ,np.int16 ,np.int32, np.int64,
                np.uint8 ,np.uint16, np.uint32, np.uint64]

        np_types = [np_type.__name__ for np_type in np_types]
        type_df = pd.DataFrame(data=np_types, columns=['class_type'])

        type_df['min_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).min)
        type_df['max_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).max)
        type_df['range'] = type_df['max_value'] - type_df['min_value']
        type_df.sort_values(by='range', inplace=True)


        for col in df.loc[:, df.dtypes <= np.integer]:
            col_min = df[col].min()
            col_max = df[col].max()
            temp = type_df[(type_df['min_value'] <= col_min) & (type_df['max_value'] >= col_max)]
            optimized_class = temp.loc[temp['range'].idxmin(), 'class_type']
            print("Название колонки : {} Минимальное значение : {} Максимальное значение : {} Можно оптимизировать к  : {}"\
                  .format(col, col_min, col_max, optimized_class))

            if inplace == 'True':
                df[col] = df[col].astype(optimized_class)

        #df.info()

        #return df
