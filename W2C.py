import pandas as pd
import gensim
import jieba
train = pd.read_csv('data/train_first.csv')
test = pd.read_csv('data/predict_first.csv')
train_line = train['Discuss'].values
test_line = test['Discuss'].values

train_words = []
test_words = []
for line in train_line:
    words = list(jieba.cut(line.strip().replace('\n','')))
    train_words.append(words)
for line in test_line:
    words = list(jieba.cut(line.strip().replace('\n','')))
    test_words.append(words)

train_words.extend(test_words)
model = gensim.models.Word2Vec(train_words, min_count=2)
model.save('glove/yun.model')

model = gensim.models.Word2Vec.load('glove/yun.model')
model.wv.save_word2vec_format('glove/yun.txt', binary=False)