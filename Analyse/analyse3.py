import pickle as pkl
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

## svm
svm = LinearSVC(C=1, loss="hinge")
svm.fit(X_train, y_train)
predict_train = svm.predict(X_train)
predict_test = svm.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

## xgboost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
predict_train = xgb.predict(X_train)
predict_test = xgb.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[Xgboost]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

## randomforest
rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
rf.fit(X_train, y_train)
predict_train = rf.predict(X_train)
predict_test = rf.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[Random Forest]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

## MLP
mlp = MLPClassifier(alpha=1, max_iter=1000)
mlp.fit(X_train, y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[Multi Linear]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

## decision tree
dt = DecisionTreeClassifier(random_state=123, max_depth=10)
dt.fit(X_train, y_train)
predict_train = dt.predict(X_train)
predict_test = dt.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[Decision Tree]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

## NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predict_train = gnb.predict(X_train)
predict_test = gnb.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[GaussianNB]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

## lightGBM
gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train)
predict_train = gbm.predict(X_train)
predict_test = gbm.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[LightGBM]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

## KNN
knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)
predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[K Nearest Neighbors]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))

# Adaboost
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
predict_train = ada.predict(X_train)
predict_test = ada.predict(X_test)

tr_score = metrics.accuracy_score(y_train, predict_train)
te_score = metrics.accuracy_score(y_test, predict_test)
print('[Adaboost]Train acc: %.4f, Test acc: %.4f.'%(tr_score, te_score))





