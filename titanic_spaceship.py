import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import MEstimateEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

class CabinTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        CabinSeries = pd.Series(X.flatten(), dtype = "string")
        CabinLocation = CabinSeries.str.split(pat = "/", expand = True)
        CabinLocation.columns = ["Deck","Number", "Side"]
        CabinLocation["Number"] = CabinLocation["Number"].astype("int32")
        return CabinLocation
    
class IdTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        for x in range(len(X[:,0])):
            X[x,0] = X[x,0][0:4]
        for x in range(len(X[:,0])):
            X[x,0] = (X[:,0] == X[x,0]).sum()
        GroupSize = X[:,0]
        
        return GroupSize
​
class SpentTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        #print(X[1])
        TotalSpent = pd.Series([sum(x) for x in X])
        TotalSpent = pd.DataFrame(TotalSpent)
        #print(TotalSpent)
        return TotalSpent
​
class Debug(BaseEstimator, TransformerMixin):
​
    def transform(self, X):
        print(X[0])
        # what other output you want
        return X
​
    def fit(self, X, y=None, **fit_params):
        return self

def read_titanic():
    train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
    test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


    X = train.copy()

    X.drop(columns = ["Name"], inplace = True)
    #X["IndexCopy"] = train.index
    X_test = test.copy()
    Y = X.pop('Transported')
    return X, Y, X_test
​
def preprocess_func():
    categorical_cols = ["HomePlanet","Destination"]
    binary_cols = ["VIP", "CryoSleep"]
    numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "VRDeck", "Spa"]
    spent_cols = ["Spa", "VRDeck","RoomService", "ShoppingMall", "FoodCourt"]
    
    categorical_transformer = Pipeline(steps = [('imputer',SimpleImputer(strategy = "most_frequent")),
                                               ('onehot', OneHotEncoder())])
    binary_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'constant', fill_value = False))])
    numerical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = "constant", fill_value = 0)),
                                              ('scaler', StandardScaler())])
    ​
    cabin_transformer = Pipeline(steps = [('imputer',SimpleImputer(strategy = "most_frequent")),
                                          ('split', CabinTransformer())])
    cabin_encoder = make_pipeline(cabin_transformer, 
                                  ColumnTransformer(transformers = 
                                                    [('onehot', OneHotEncoder(), ["Deck", "Side"])],
                                  remainder = "passthrough"))
    spent_transformer = make_pipeline(SimpleImputer(strategy = "constant", fill_value = 0),
                                      SpentTransformer(), StandardScaler())
    ​
    id_transformer = make_pipeline(SimpleImputer(strategy = "most_frequent"), IdTransformer(),
                                   MEstimateEncoder())
    debug_transformer = make_pipeline(Debug())
    ​
    preprocessor = ColumnTransformer(transformers = [('cat', categorical_transformer, categorical_cols),
                                                     ('num', numerical_transformer, numerical_cols),
                                                     ('cabin', cabin_encoder, ["Cabin"]),
                                                     ('binary', binary_transformer, binary_cols),
                                                     ('Id', id_transformer, ["PassengerId"]),
                                                     ('spent', spent_transformer, spent_cols)], remainder = "drop")
​
​    return preprocessor
​

def model_pipeline():
    classifier = RandomForestClassifier(n_estimators = 250,max_leaf_nodes = 1000, random_state = 0)
    my_pipeline = Pipeline(steps = [('preprocessor', preprocess_func()), ('model', classifier)])
    return my_pipeline
    
def score_model(X,Y):
    my_pipeline = model_pipeline()
    scores = cross_val_score(my_pipeline, X, Y, cv = 5)
    return scores.mean()

def grid_search(X,Y):
    classifier = RandomForestClassifier(n_estimators = 250,max_leaf_nodes = 1000, random_state = 0)
    my_pipeline = Pipeline(steps = [('preprocessor', preprocess_func()), ('model', classifier)])
    parameters = {'model__n_estimators' : [100,250,500], 'model__max_leaf_nodes' : [500,1000,2000]}
    ​
    grid = GridSearchCV(estimator = my_pipeline, param_grid = parameters)
    grid.fit(X,Y)
    ​
    print(grid.best_params_)
    print(grid.best_score_)
    return None

def titanic_submit(X,Y,X_test)
    my_pipeline = model_pipeline()
    my_pipeline.fit(X, Y)
    preds = my_pipeline.predict(X_test)
    
    output = pd.DataFrame({'PassengerId': X_test["PassengerId"],
                       'Transported': preds})
    output.to_csv('submission.csv', index=False)
    return None

def main():
    X, Y, X_test = read_titanic()
    score = score_model(X,Y)
    print (score)

if __name__ == "__main__":
    main()
