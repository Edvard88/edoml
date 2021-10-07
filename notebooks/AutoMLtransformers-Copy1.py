import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
#from sklearn.impute import SimpleImputer
#from sklearn.pipeline import FeatureUnion, Pipeline 



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
            print("Col name : {} Col min_value : {} Col max_value : {} Optimized Class : {}"\
                  .format(col, col_min, col_max, optimized_class))

            if inplace == 'True':
                df[col] = df[col].astype(optimized_class)

        #df.info()

        #return df

        
        
class AutoMlClassification(BaseAutoMlEstimator):    
    
    def __init__(self, df, X_train, X_test, y_train, y_test, target_col, reports_path='/reports'):
        super().__init__(df, X_train, X_test, y_train, y_test, target_col, reports_path='/reports')
        
    
    def get_report_about_target(self, df, target_col ,reports_path='/reports'):
        
        ### Формируем отчет по таргету DF
        
        target_report = df.groupby(target_col).count().reset_index()

        dict_count_traget = df[target_col].value_counts().to_dict()
        dict_count_traget_norm = df[target_col].value_counts(normalize=True).to_dict()

        target_report['count'] = target_report[target_col].map(dict_count_traget)
        target_report['count_norm'] = target_report[target_col].map(dict_count_traget_norm)
        
        target_report.to_csv('..'+ reports_path + '/target_report.csv', index=False)    
        
        print("Отчет по статистикам по таргету сформирован")
        return target_report
    
    
    
    def fit_report(self, X_train, X_test, y_train, y_test, reports_path='/reports'):
        
        
        classifiers = [LogisticRegression,
               KNeighborsClassifier,
               #GradientBoostingClassifier(), 
               RandomForestClassifier] 
#               SVC()] # 

        classifiers_name = ['LogisticRegression',

                            'KNeighborsClassifier',
                            #'GradientBoostingClassifier', 
                            'RandomForestClassifier'] 
        #                    'SVC']
        
        
        # Настройка параметров выбранных алгоритмов с помощью GridSearchCV 
        n_folds = 5
        scores = []
        fits = []
        logistic_params = {'penalty': ('l1', 'l2'),
                           'C': (.01,.1,1,5)}

        knn_params = {'n_neighbors': list(range(3, 6, 2))}


        gbm_params = {'n_estimators': [100, 300, 500],
                      'learning_rate':(0.1, 0.5, 1),
                      'max_depth': list(range(3, 6)), 
                      'min_samples_leaf': list(range(10, 31, 10))}



        forest_params = {'n_estimators': [10, 30, 50],
                         'criterion': ('gini', 'entropy')}

        #svm_param = {'kernel' : ('linear', 'rbf'), 'C': (.5, 1, 2)} - очень долго считал
        #params = [logistic_params, knn_params, gbm_params, forest_params]

        params = [logistic_params, knn_params ,forest_params]        
        
        
        np.random.seed(0)

        df1 = pd.DataFrame()

        skf = StratifiedKFold(n_splits=2, random_state=0)

        for i , each_classifier in enumerate(classifiers):
            clf = each_classifier
            clf_params = params[i]
            clf_classifiers_name = classifiers_name[i]
            print("classifiers_name", clf_classifiers_name)

            for tmp_params in self.get_params_combination(clf_params):
                print("Параметры: ", tmp_params)
                skf_index = skf.split(X_train, y_train)
                for fold, (train_idx, test_idx) in enumerate(skf_index):
                    print("Размер тренировочного / тестового датасета: ", len(train_idx), len(test_idx))

                    # Формируем тренеровочный и валидационный датасет
                    X_train_tmp, X_test_tmp = X_train.iloc[train_idx], X_train.iloc[test_idx]
                    y_train_tmp, y_test_tmp = y_train.iloc[train_idx], y_train.iloc[test_idx]

                    # Получаем модель
                    tmp_clf = clf(**tmp_params)

                    # Замеряем время fit
                    start_time = time.time()
                    pred = tmp_clf.fit(X_train_tmp, y_train_tmp)
                    fit_time = time.time() - start_time


                    # Замеряем время predict
                    start_time = time.time()
                    pred = tmp_clf.predict(X_test_tmp)
                    predict_time = time.time() - start_time

                    tmp_params_string = ", ".join(("{}={}".format(*i) for i in tmp_params.items()))

                    data = {'model_name' : clf_classifiers_name, 
                            'fold' : fold,
                            'params' : tmp_params_string,
                            'fit_time' : fit_time, 
                            'predict_time' : predict_time,
                            'roc_auc':roc_auc_score(y_test_tmp, pred)

                    }

                    # Расширяем другими параметрами
                    data.update(tmp_params)

                    # Формируем финальный датафрейм
                    df1 = df1.append(data, ignore_index=True)

                    
        df1.to_csv('..'+ reports_path + '/model_report.csv', index=False)  
        print("Отчет по модели сформирован")

    