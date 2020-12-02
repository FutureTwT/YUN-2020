import pandas as pd
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import scipy
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix, hstack
import re
import numpy as np
def get_data():
    train = pd.read_csv('data/train_first.csv')
    test = pd.read_csv('data/predict_first.csv')
    # upsampling
    '''
    df1 = train[train['Score'] == 1]
    for i in range(20):
        train = pd.concat([train, df1], axis=0)
    
    df2 = train[train['Score'] == 2]
    for i in range(5):
        train = pd.concat([train, df2], axis=0)
    df3 = train[train['Score'] == 3]
    for i in range(2):
        train = pd.concat([train, df3], axis=0)
    '''

    train = train.sample(frac=1).reset_index(drop=True)


    data = pd.concat([train, test])
    # print(train['Score'].max())
    # print(train['Score'].min())
    print('train %s test %s'%(train.shape,test.shape))
    print('train columns',train.columns)
    return data,train.shape[0],train['Score'],test['Id']

def clear_str(string):
    string = re.sub(r'[0-9a-zA-Z]+', '', string)
    string = string.replace('，','').replace('。','').replace('～','').replace(' ','').replace('！','').\
        replace('<br/>','').replace('；','').replace('）','').replace('（','').replace('.','').\
        replace('“','').replace('”','').replace(',','').replace('【','').replace('】','').replace('~','').\
        replace('\xa0','').replace('《','').replace('》','').replace('<','').replace('>','').replace('/','').strip()
    cut_str = jieba.cut(string.strip())
    list_str = [word for word in cut_str]
    string = ' '.join(list_str)
    return string

# 分词处理
def split_discuss(data):
    data['length'] = data['Discuss'].apply(lambda x:len(x))
    data['Discuss'] = data['Discuss'].apply(lambda x: clear_str(x))
    return data

# 预处理
def pre_process():
    data,train_line,Y,test_id = get_data()
    data = split_discuss(data)
    cv = CountVectorizer(ngram_range=(1,2))
    discuss = cv.fit_transform(data['Discuss'])
    tf = TfidfVectorizer(max_df=10000,ngram_range=(1,2))
    discuss_tf = tf.fit_transform(data['Discuss'])
    # length = csr_matrix(pd.get_dummies(data['length'],sparse=True).values)
    data = hstack((discuss,discuss_tf)).tocsr()
    return data[:train_line],data[train_line:],Y,test_id

def xx_mse_s(y_true,y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res':list(y_pre)})

    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / ( 1 + mean_squared_error(y_true,y_pre['res'].values)**0.5)


X_train_all,X_test_all,Y_train_all,test_id = pre_process()

kf = KFold(n_splits=3,shuffle=True,random_state=42)
kf = kf.split(X_train_all)

# get model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import Ridge, LogisticRegression
from mlxtend.classifier import StackingCVClassifier

# DT = tree.DecisionTreeClassifier()
Ri = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=250, normalize=False, tol=0.01)
RF = RandomForestClassifier(random_state=10)
EXT = ExtraTreesClassifier()
LR = LogisticRegression()
SCLF = StackingCVClassifier(classifiers=[Ri, RF, EXT], meta_classifier=LR, use_probas=True, verbose=3)

cv_pred_EXT = []
cv_pred_Ri = []
cv_pred_RF = []
cv_pred_LR = []

xx_mse = []
for i ,(train_fold,test_fold) in enumerate(kf):
    print(type(X_train_all))
    X_train, X_validate, Y_train, Y_validate = X_train_all[train_fold, :], X_train_all[test_fold, :], Y_train_all[train_fold], Y_train_all[test_fold]

    EXT.fit(X_train, Y_train)
    Ri.fit(X_train, Y_train)
    RF.fit(X_train, Y_train)
    LR.fit(X_train, Y_train)

    pred_EXT = EXT.predict(X_validate)
    pred_Ri = Ri.predict(X_validate)
    pred_RF = RF.predict(X_validate)
    pred_LR = LR.predict(X_validate)

    print('EXT: ', xx_mse_s(Y_validate, pred_EXT))
    print('Ri: ', xx_mse_s(Y_validate, pred_Ri))
    print('RF: ', xx_mse_s(Y_validate, pred_RF))
    print('LR: ', xx_mse_s(Y_validate, pred_LR))

    cv_pred_EXT.append(EXT.predict(X_test_all))
    cv_pred_Ri.append(Ri.predict(X_test_all))
    cv_pred_RF.append(RF.predict())
    xx_mse.append(xx_mse_s(Y_validate, pred))

import numpy as np
print('xx_result',np.mean(xx_mse))

s = 0
for i in cv_pred:
    s = s + i
    print(i)

s = s / 3
s = list(s)
res = pd.DataFrame()
res['Id'] = list(test_id)
res['pre'] = [round(float(num)) for num in s]

res.to_csv('t_20180215_1.csv',index=False,header=False)

