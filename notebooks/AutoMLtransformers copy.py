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
        
        ### ?????????????????? ?????????? ???? ?????????? DF
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
        
        print("?????????? ???? ?????????????????????? ???? ?????????? ??????????????????????")
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
            print("???????????????? ?????????????? : {} ?????????????????????? ???????????????? : {} ???????????????????????? ???????????????? : {} ?????????? ???????????????????????????? ??  : {}"\
                  .format(col, col_min, col_max, optimized_class))

            if inplace == 'True':
                df[col] = df[col].astype(optimized_class)

        #df.info()

        #return df

        
        
class AutoMlClassification(BaseAutoMlEstimator):    
    
    def __init__(self, df, X_train, X_test, y_train, y_test, target_col, reports_path='/reports'):
        super().__init__(df, X_train, X_test, y_train, y_test, target_col, reports_path='/reports')
        
    
    def get_report_about_target(self, df, target_col ,reports_path='/reports'):
        
        ### ?????????????????? ?????????? ???? ?????????????? DF
        
        target_report = df.groupby(target_col).count().reset_index()

        dict_count_traget = df[target_col].value_counts().to_dict()
        dict_count_traget_norm = df[target_col].value_counts(normalize=True).to_dict()

        target_report['count'] = target_report[target_col].map(dict_count_traget)
        target_report['count_norm'] = target_report[target_col].map(dict_count_traget_norm)
        
        target_report.to_csv('..'+ reports_path + '/target_report.csv', index=False)    
        
        print("?????????? ???? ?????????????????????? ???? ?????????????? ??????????????????????")
        return target_report
    
    
    
    def fit_report(self, X_train, X_test, y_train, y_test, reports_path='/reports'):
        
        digits_features = X_train.select_dtypes(include=['number']).columns.values.tolist()
        cat_features = X_train.select_dtypes(include=['object', ]).columns.values.tolist()
        
        classifiers = [LogisticRegression,
               #KNeighborsClassifier,
               #GradientBoostingClassifier(), 
               RandomForestClassifier] 
        #               SVC()] # 

        classifiers_name = ['LogisticRegression',

                            #'KNeighborsClassifier',
                            #'GradientBoostingClassifier', 
                            'RandomForestClassifier'] 
        #                    'SVC']


        # ?????????????????? ???????????????????? ?????????????????? ???????????????????? ?? ?????????????? GridSearchCV 
        n_folds = 5
        scores = []
        fits = []
        logistic_params = {'penalty': ('l1', 'l2'),
                           'C': (.01,5)}

        knn_params = {'n_neighbors': list(range(3, 6, 2))}


        gbm_params = {'n_estimators': [100, 300, 500],
                      'learning_rate':(0.1, 0.5, 1),
                      'max_depth': list(range(3, 6)), 
                      'min_samples_leaf': list(range(10, 31, 10))}



        forest_params = {'n_estimators': [10, 30],
                         'criterion': ('gini', 'entropy')}

        #svm_param = {'kernel' : ('linear', 'rbf'), 'C': (.5, 1, 2)} - ?????????? ?????????? ????????????
        #params = [logistic_params, knn_params, gbm_params, forest_params]

        params = [logistic_params, forest_params]        



        ############# ???????????????????? NaN #############
        imputer_numeric_list = [ModifiedSimpleImputer]
        imputer_numeric_name = ['ModifiedSimpleImputer'] 

        simpleimputer_params_numeric = {'fill_value': [0, -1],
                                        'strategy': ['constant']}
        params_numeric_list = [simpleimputer_params_numeric]        





        imputer_cat_list = [ModifiedSimpleImputer]
        imputer_cat_name = ['ModifiedSimpleImputer'] 

        simpleimputer_params_cat = {'strategy': ['constant'],
                                   'fill_value': ['missing']}
        params_cat_list = [simpleimputer_params_cat]        

        ############# ???????????????????? NaN #############



        ############# Scaler ################

        scaler_list = [MinMaxScaler]
        scaler_name_list = ['MinMaxScaler'] 

        scaler_params = {'feature_range': [(0,1) , (2,3)]}
        params_scaler_list = [scaler_params]        


        #####################################


        np.random.seed(0)




        np.random.seed(0)

        df1 = pd.DataFrame()

        skf = StratifiedKFold(n_splits=3, random_state=0)


        # for i , each_imputer_cat in enumerate(imputer_cat_list):
        #     imputer_cat = each_imputer_cat
        #     imputer_cat_name = imputer_cat_name[i]
        #     imputer_cat_params = params_cat_list[i]
        #     print("imputer_cat_name", imputer_cat_name)    
        #     for tmp_imputer_cat_params in get_params_combination(imputer_cat_params):
        #         print("??????????????????: ", tmp_imputer_cat_params)


        for i , each_imputer_numeric in enumerate(imputer_numeric_list):
            imputer_numeric = each_imputer_numeric
            imputer_numeric_name = imputer_numeric_name[i]
            imputer_numeric_params = params_numeric_list[i]
            print("imputer_numeric_name fill_na", imputer_numeric_name)    
            for current_imputer_numeric_params in self.get_params_combination(imputer_numeric_params):
                print('\n', "??????????????????: ", current_imputer_numeric_params)


                for i , each_scaler in enumerate(scaler_list):
                    scaler = each_scaler
                    scaler_name = scaler_name_list[i]
                    scaler_params = params_scaler_list[i]
                    print("scaler_name", scaler_name,)    
                    for current_scaler_param in self.get_params_combination(scaler_params):
                        print('\n', "?????????????????? scaler: ", current_scaler_param, '\n')

                        for i , each_classifier in enumerate(classifiers):
                            clf = each_classifier
                            clf_params = params[i]
                            clf_classifiers_name = classifiers_name[i]
                            print("classifiers_name", clf_classifiers_name,)

                            for tmp_params in self.get_params_combination(clf_params):
                                print("?????????????????? classifiers: ", tmp_params)
                                skf_index = skf.split(X_train, y_train)
                                for fold, (train_idx, test_idx) in enumerate(skf_index):
                                    print("???????????? ???????????????????????????? / ?????????????????? ????????????????: ", len(train_idx), len(test_idx))

                                    # ?????????????????? ?????????????????????????? ?? ?????????????????????????? ??????????????
                                    X_train_tmp, X_test_tmp = X_train.iloc[train_idx], X_train.iloc[test_idx]
                                    y_train_tmp, y_test_tmp = y_train.iloc[train_idx], y_train.iloc[test_idx]

                                    # ???????????????? ????????????
                                    #tmp_clf = clf(**tmp_params)

                                    tmp_clf = Pipeline([
                                            # Use FeatureUnion to combine the features
                                            ('union', ModifiedFeatureUnion(
                                                transformer_list=[
                                                     # categorical features
                                                    ('categorical', Pipeline([
                                                         ('selector', FeatureSelector(columns = cat_features)),
                                                         ('imputer', ModifiedSimpleImputer(strategy='constant', fill_value='missing')),
                                                         ('label_encoding', MyLEncoder())
                                                    ])),
                                                    # numeric features
                                                    ('numeric', Pipeline([
                                                         ('selector', FeatureSelector(columns = digits_features)),
                                                         ('imputer', imputer_numeric(**current_imputer_numeric_params)),
                                                         ('scaler', scaler(**current_scaler_param))
                                                    ])),
                                                ])),
                                            # Use model fit
                                            ('model_fitting', clf(**tmp_params)),
                                        ])



                                    # ???????????????? ?????????? fit
                                    start_time = time.time()
                                    pred = tmp_clf.fit(X_train_tmp, y_train_tmp)
                                    fit_time = time.time() - start_time


                                    # ???????????????? ?????????? predict
                                    start_time = time.time()
                                    pred = tmp_clf.predict(X_test_tmp)
                                    predict_time = time.time() - start_time

                                    clf_tmp_params_string = ", ".join(("{}={}".format(*i) for i in tmp_params.items()))
                                    scale_tmp_params_string = ", ".join(("{}={}".format(*i) for i in current_scaler_param.items()))
                                    imputer_numeric_params_string = ", ".join(("{}={}".format(*i) for i in current_imputer_numeric_params.items()))


                                    data = {'classifier_name' : clf_classifiers_name,
                                            'classifier_params' : clf_tmp_params_string,
                                            'scaler_name': scaler_name,
                                            'scaler_params': scale_tmp_params_string,

                                            'imputer_name': imputer_numeric_name,
                                            'imputer_params': imputer_numeric_params_string,

                                            'fold' : fold,
                                            'fit_time' : fit_time, 
                                            'predict_time' : predict_time,
                                            'roc_auc':roc_auc_score(y_test_tmp, pred)

                                    }

                                    # ?????????????????? ?????????????? ??????????????????????
                                    data.update(tmp_params) # ?????????????????? ????????????????????????????
                                    data.update(current_scaler_param) # ?????????????????? scale
                                    data.update(current_imputer_numeric_params)

                                    # ?????????????????? ?????????????????? ??????????????????
                                    df1 = df1.append(data, ignore_index=True)

                    
        df1.to_csv('..'+ reports_path + '/model_report.csv', index=False)  
        print("?????????? ???? ???????????? ??????????????????????")

    