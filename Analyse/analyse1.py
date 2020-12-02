import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import jieba

train = pd.read_csv('data/train_first.csv')
# test = pd.read_csv('data/predict_first.csv')

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

def split_discuss(data):
    data['length'] = data['Discuss'].apply(lambda x:len(x))
    data['Discuss'] = data['Discuss'].apply(lambda x: clear_str(x))
    return data

data = split_discuss(train)
print(sum(data['length']))
# print(data['length'])
# print(max(data['length']), min(data['length']))
# plt.plot(range(1, 100001), data['length'])
# plt.xlabel('Sample')
# plt.ylabel('Length of discuss')
# plt.savefig('fig/length.png')

