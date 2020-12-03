import jieba
import pandas as pd
# import gensim
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('../data/train_first.csv')
train_line = train['Discuss'].values

dict = {}
for idx, line in enumerate(train_line):
    words = list(jieba.cut(line.strip().replace('\n','')))
    # print(len(words))
    for w in words:
        if w not in dict.keys():
            dict[w] = 1
        else:
            dict[w] += 1
from wordcloud import WordCloud

wordcloud = WordCloud(font_path='../data/simhei.ttf',
                      scale=32,
                      mode='RGBA',
                      background_color='white')
wc = wordcloud.fit_words(dict)

plt.imshow(wc)
plt.axis('off')
plt.savefig('../fig/wc.png')







