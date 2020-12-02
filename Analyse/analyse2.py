import pandas as pd
import numpy as np
import os
from keras import backend
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers
from keras.engine.topology import Layer
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import pickle as pkl

dict_size = 57323
num_labels = 5

train = pd.read_csv('../data/train_first.csv')
# title = train['Discuss']
title = pkl.load(open('../data/train_sequence.pkl', 'rb'))
label = train['Score']
X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

# process the label
y_train = y_train.values - 1
y_test = y_test.values - 1
y1 = np.zeros(shape=(y_train.shape[0], 5))
y2 = np.zeros(shape=(y_test.shape[0], 5))
y1[np.arange(0, y1.shape[0]), y_train] = 1
y2[np.arange(0, y2.shape[0]), y_test] = 1
y_train = y1
y_test = y2

# Match the input format of the model
x_train_padded_seqs = pad_sequences(X_train, maxlen=20)
x_test_padded_seqs = pad_sequences(X_test, maxlen=20)


## LSTM
tbCallBack = TensorBoard(log_dir='../log/LSTM2')
model = Sequential()
model.add(Embedding(dict_size + 1, 256, input_length=20))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=512,
          epochs=30,
          shuffle=True,
          callbacks=[tbCallBack],
          validation_data=(x_test_padded_seqs, y_test))
model.save('model/LSTM.model')
del model

## CNN
tbCallBack = TensorBoard(log_dir='../log/CNN')
model = Sequential()
model.add(Embedding(dict_size + 1, 256, input_length=20))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(256, 3, padding='same'))
model.add(MaxPool1D(3, 3, padding='same'))
model.add(Convolution1D(128, 3, padding='same'))
model.add(MaxPool1D(3, 3, padding='same'))
model.add(Convolution1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=512,
          epochs=10,
          shuffle=True,
          callbacks=[tbCallBack],
          validation_data=(x_test_padded_seqs, y_test))
model.save('model/CNN.model')
del model

## TextCNN
tbCallBack = TensorBoard(log_dir='../log/TextCNN')
main_input = Input(shape=(20,), dtype='float64')
embedder = Embedding(dict_size + 1, 300, input_length=20)
embed = embedder(main_input)
cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
cnn3 = MaxPool1D(pool_size=4)(cnn3)
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
main_output = Dense(num_labels, activation='softmax')(drop)
model = Model(inputs=main_input, outputs=main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=512,
          epochs=10,
          shuffle=True,
          callbacks=[tbCallBack],
          validation_data=(x_test_padded_seqs, y_test))
model.save('model/TextCNN.model')
del model

tbCallBack = TensorBoard(log_dir='../log/CNN-LSTM')
main_input = Input(shape=(20,), dtype='float64')
embed = Embedding(dict_size + 1, 256, input_length=20)(main_input)
cnn = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
cnn = MaxPool1D(pool_size=4)(cnn)
cnn = Flatten()(cnn)
cnn = Dense(256)(cnn)
rnn = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(embed)
rnn = Dense(256)(rnn)
con = concatenate([cnn, rnn], axis=-1)
main_output = Dense(num_labels, activation='softmax')(con)
model = Model(inputs=main_input, outputs=main_output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train_padded_seqs, y_train,
          batch_size=512,
          epochs=10,
          shuffle=True,
          callbacks=[tbCallBack],
          validation_data=(x_test_padded_seqs, y_test))
model.save('model/CNN_LSTM.model')
del model
