import jieba
import pandas as pd
# import gensim
import pickle as pkl
import numpy as np
'''
dict = {}
with open('../glove/yun.txt', 'r') as file:
    for idx, line in enumerate(file.readlines()):
        if idx == 0:
            continue
        words = line.split()
        if words[0] not in dict.keys():
            dict[words[0]] = idx
    print(len(dict.keys()))# 57323
    # print(dict)
'''
dict = pkl.load(open('../glove/embedding.pkl', 'rb'))
print(len(dict.keys()))

# train_words = []

train = pd.read_csv('../data/train_first.csv')
train_line = train['Discuss'].values

train_embedding = np.zeros(shape=(train_line.shape[0], 100))

for idx, line in enumerate(train_line):
    words = list(jieba.cut(line.strip().replace('\n','')))
    # print(len(words))
    # tmp = []
    cnt = 0
    for w in words:
        if w in dict.keys():
            # tmp.append(dict[w])
            cnt += 1
            train_embedding[idx] = train_embedding[idx] + dict[w]
    if cnt != 0:
        train_embedding[idx] = train_embedding[idx] / cnt

    # train_words.append(tmp)

'''
print(len(train_words), len(train_words[0]), train_words[0])

with open('../data/train_sequence.pkl', 'wb') as file:
    import pickle as pkl
    pkl.dump(train_words, file)
'''

with open('../data/train_embedding.pkl', 'wb') as file:
    pkl.dump(train_embedding, file)