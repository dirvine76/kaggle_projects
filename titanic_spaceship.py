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

train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
​
​
X = train.copy()
​
X.drop(columns = ["Name"], inplace = True)
#X["IndexCopy"] = train.index
X_test = test.copy()
Y = X.pop('Transported')
​
categorical_cols = [cname for cname in X.columns if (X[cname].dtype == "object")]
categorical_cols = ["HomePlanet","Destination"]
binary_cols = ["VIP", "CryoSleep"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ["int64","float64"]]
spent_cols = ["Spa", "VRDeck","RoomService", "ShoppingMall", "FoodCourt"]
​
​
#estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
#combined = FeatureUnion(estimators)
​
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, 
                                                      train_size=0.75, test_size=0.25,
                                                      random_state=0)

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
​
​

classifier = RandomForestClassifier(n_estimators = 250,max_leaf_nodes = 1000, random_state = 0)
#classifier = CatBoostClassifier(verbose = 0)
#classifier = xgb.XGBClassifier(seed=42, gamma = 0.2, learning_rate =  0.1, max_depth =  8, min_child_weight = 1)
#classifier = GaussianProcessClassifier()
#classifier = LogisticRegression(max_iter = 400)
my_pipeline = Pipeline(steps = [('preprocessor', preprocessor), ('model', classifier)])
#my_pipeline.fit(X_train, Y_train)
#preds = my_pipeline.predict(X_test)
​
scores = cross_val_score(my_pipeline, X, Y, cv = 5)
#parameters = {'model__n_estimators' : [100,250,500], 'model__max_leaf_nodes' : [500,1000,2000]}
​
#grid = GridSearchCV(estimator = my_pipeline, param_grid = parameters)
#grid.fit(X,Y)
​
#print(grid.best_params_)
#print(grid.best_score_)

print(scores.mean())
