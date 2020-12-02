import gensim
import os
import numpy as np

# load glove word embedding data
GLOVE_DIR = "../glove/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'yun.txt'), encoding='utf-8')
for num, line in enumerate(f):
    if num == 0:
        continue
    # print(line)
    values = line.split()
    word = values[0]
    if word not in embeddings_index.keys():
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

f.close()

import pickle as pkl
with open('../glove/embedding.pkl', 'wb') as file:
    pkl.dump(embeddings_index, file)
