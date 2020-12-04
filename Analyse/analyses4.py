import pickle as pkl
import numpy as np
import pandas as pd
from utils import *

from sklearn import datasets
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

'''
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
'''

X = pkl.load(open('../data/train_embedding.pkl', 'rb'))
train = pd.read_csv('../data/train_first.csv')
Y = train['Score']
# print(X.shape, Y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
cv = StratifiedShuffleSplit(X.shape[0], test_size=0.1, random_state=0)

'''
## svm
svm = LinearSVC(C=1, loss="hinge")
title = 'SVM'
plt = plot_learning_curve(svm, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/svm.png')
'''

## xgboost
xgb = XGBClassifier()
title = 'XGBoost'
plt = plot_learning_curve(xgb, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/XGBoost.png')

## randomforest
rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
title = 'RandomForest'
plt = plot_learning_curve(rf, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/RandomForest.png')

## MLP
mlp = MLPClassifier(alpha=1, max_iter=1000)
title = 'MLP'
plt = plot_learning_curve(mlp, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/mlp.png')

## decision tree
dt = DecisionTreeClassifier(random_state=123, max_depth=10)
title = 'DecisionTree'
plt = plot_learning_curve(dt, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/DecisionTree.png')

## NB
gnb = GaussianNB()
title = 'GaussianNB'
plt = plot_learning_curve(gnb, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/gnb.png')

## lightGBM
gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20)
title = 'LightGBM'
plt = plot_learning_curve(gbm, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/LightGBM.png')

## KNN
knn = KNeighborsClassifier(5)
title = 'KNN'
plt = plot_learning_curve(knn, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/knn.png')

# Adaboost
ada = AdaBoostClassifier()
title = 'AdaBoost'
plt = plot_learning_curve(ada, title, X, Y, (0.0, 1.01), cv=cv, n_jobs=8)
plt.savefig('../fig/AdaBoost.png')


