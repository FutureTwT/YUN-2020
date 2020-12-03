import gensim
import os
import numpy as np
import seaborn as sns

# load glove word embedding data
GLOVE_DIR = "../glove/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'yun.txt'), encoding='utf-8')
embed_data = np.ndarray(shape=(57323, 100))

for num, line in enumerate(f):
    if num == 0:
        continue
    # print(line)
    values = line.split()
    word = values[0]
    if word not in embeddings_index.keys():
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        if coefs.shape[0] != 100:
            # TODO: There is a bug.
            tmp = np.zeros(shape=(100))
            tmp[:coefs.shape[0]] = coefs
            embed_data[num-1, :] = tmp
        else:
            embed_data[num-1, :] = coefs
f.close()

print(len(embeddings_index.keys()))

corr = np.corrcoef(embed_data[:, :10].T)
print(corr)
fig = sns.heatmap(corr)
fig.get_figure().savefig('../fig/heatmap.png')

'''
import pickle as pkl
with open('../glove/embedding.pkl', 'wb') as file:
    pkl.dump(embeddings_index, file)
'''