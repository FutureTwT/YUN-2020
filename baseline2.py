import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
import jieba
import re
from sklearn.model_selection import StratifiedKFold, KFold
import random
import fasttext

# ID Discuss Score
train_data_path = 'data/train_first.csv'
test_data_path = 'data/predict_first.csv'
train_df = pd.read_csv(train_data_path, header=0, encoding='utf-8')
test_df = pd.read_csv(test_data_path, header=0, encoding='utf-8')

train_df.drop_duplicates(subset='Discuss', keep='first', inplace=True)
stop_word = []
stop_word_path = 'stopWordList'


def clear_str(string):
    string = re.sub(r'[0-9a-zA-Z]+', '', string)
    string = string.replace('，', '').replace('。', '').replace('～', '').replace(' ', '').replace('！', ''). \
        replace('<br/>', '').replace('；', '').replace('）', '').replace('（', '').replace('.', '').replace('？',''). \
        replace('“', '').replace('”', '').replace(',', '').replace('【', '').replace('】', '').replace('~', ''). \
        replace('\xa0', '').replace('《', '').replace('》', '').replace('<','').replace('>','').replace('/','').\
        replace('-','').replace('：','').replace('…','').strip()
    cut_str = jieba.cut(string.strip())
    list_str = [word for word in cut_str]
    string = ' '.join(list_str)
    return string


# Clear all lines in  train and test
train_df['Discuss'] = train_df['Discuss'].map(lambda x: clear_str(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: clear_str(x))


def fillnull(x):
    if x == '':
        return '空白'
    else:
        return x


# Fill all the null value in train and test
train_df['Discuss'] = train_df['Discuss'].map(lambda x: fillnull(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: fillnull(x))

# Count the score
Score_5_id = train_df[train_df['Score'] == 5].index.tolist()
Score_4_id = train_df[train_df['Score'] == 4].index.tolist()
Score_3_id = train_df[train_df['Score'] == 3].index.tolist()
Score_2_id = train_df[train_df['Score'] == 2].index.tolist()
Score_1_id = train_df[train_df['Score'] == 1].index.tolist()

print('Score 5: ', len(Score_5_id))
print('Score 4: ', len(Score_4_id))
print('Score 3: ', len(Score_3_id))
print('Score 2: ', len(Score_2_id))
print('Score 1: ', len(Score_1_id))


# Split data
def split_sample(sample, n=4):
    num_sample = len(sample)
    sub_length = int(1 / n * num_sample)
    sub_sample = []
    for i in range(n):
        sub = [i * sub_length, (i + 1) * sub_length]
        sub_sample.append(sub)
    return sub_sample


Score_5_sample = split_sample(Score_5_id)
Score_4_sample = split_sample(Score_4_id)
Score_3_sample = split_sample(Score_3_id)
# Score_2_sample = split_sample(Score_2_id)
# Score_1_sample = split_sample(Score_1_id)

df1_index = [Score_5_sample[0], Score_4_sample[0], Score_3_sample[0], Score_2_id, Score_1_id]
df1_index = [_ for sample in df1_index for _ in sample]
random.shuffle(df1_index)

df2_index = [Score_5_sample[1], Score_4_sample[1], Score_3_sample[1], Score_2_id, Score_1_id]
df2_index = [_ for sample in df2_index for _ in sample]
random.shuffle(df2_index)

df3_index = [Score_5_sample[2], Score_4_sample[2], Score_3_sample[2], Score_2_id, Score_1_id]
df3_index = [_ for sample in df3_index for _ in sample]
random.shuffle(df3_index)

df4_index = [Score_5_sample[3], Score_4_sample[3], Score_3_sample[3], Score_2_id, Score_1_id]
df4_index = [_ for sample in df4_index for _ in sample]
random.shuffle(df4_index)

df1 = train_df.loc[df1_index, :]
df1 = df1.sample(frac=1)

df2 = train_df.loc[df2_index, :]
df2 = df2.sample(frac=1)

df3 = train_df.loc[df3_index, :]
df3 = df3.sample(frac=1)

df4 = train_df.loc[df4_index, :]
df4 = df4.sample(frac=1)


def fasttext_data(data, label):
    fasttext_data = []
    for i in range(len(label)):
        sent = str(data[i]) + '\t__label__' + str(int(label[i]))
        fasttext_data.append(sent)
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        for data in fasttext_data:
            f.write(data)
            f.write('\n')
    return 'data/train.txt'


def get_predict(pred):
    score = np.array([1, 2, 3, 4, 5])
    pred2 = []
    for p in pred:
        pr = np.sum(p * score)
        pred2.append(pr)
    return np.array(pred2)


def rmse(true_label, pred):
    true_label = np.array(true_label)
    pred = np.array(pred)
    n = len(true_label)
    dist = true_label - pred
    rmse = np.sqrt(np.sum(dist ** 2) / n)
    return 1 / (1 + rmse)


def fast_cv(df):
    df = df.reset_index(drop=True)
    df['Score'].fillna(5, inplace=True)
    X = df['Discuss'].values
    Y = df['Score'].values
    print(np.where(np.isnan(Y)))

    fast_pred = []
    folds = list(KFold(n_splits=5, shuffle=True, random_state=2018).split(X, Y))
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        print(X_test)
        train_file = fasttext_data(X_train, Y_train)
        classifier = fasttext.supervised(train_file, 'model.model', lr=0.01, dim=128, label_prefix='__label__')
        result = classifier.predict_proba(X_test.tolist(), k=5)

        pred = [[int(sco) * proba for sco, proba in result_i] for result_i in result]
        pred = [sum(pred_i) for pred_i in pred]
        print(rmse(Y_test, pred))

        test_result = classifier.predict_proba(test_df['Discuss'].tolist(), k=5)
        print('predict ending...')
        fast_predi = [[int(sco) * proba for sco, proba in result_i] for result_i in test_result]
        fast_predi = [sum(pred_i) for pred_i in fast_predi]
        fast_pred.append(fast_predi)

    fast_pred = np.array(fast_pred)
    fast_pred = np.mean(fast_pred, axis=0)
    return fast_pred


test_pred1 = fast_cv(df1)
test_pred2 = fast_cv(df2)
test_pred3 = fast_cv(df3)
test_pred4 = fast_cv(df4)
test_pred5 = fast_cv(train_df)

data = np.zeros((len(test_df), 5))
sub_df = pd.DataFrame(data)
sub_df.columns = ['Id', 'fast1', 'fast2', 'fast3', 'fast4', 'all']
sub_df['Id'] = test_df['Id'].values
sub_df['fast1'] = test_pred1
sub_df['fast2'] = test_pred2
sub_df['fast3'] = test_pred3
sub_df['fast4'] = test_pred4
sub_df['all'] = test_pred5

sub_df['mean'] = sub_df.mean(axis=1)
test_pred = fast_cv(test_df)
sub_df['mean2'] = test_pred
print(sub_df.describe())

pred = sub_df['mean2'].values
pred = np.where(pred > 4.7, 5, pred)
sub_df['mean2'] = pred

sub_df[['Id', 'mean2']].to_csv('fastsub2.csv', header=None, index=False)
