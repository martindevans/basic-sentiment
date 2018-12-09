import utils

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import os

def load_data():
    filenames = [
        "neutral.tsv",
        "acllmdb-negative.tsv",
        "acllmdb-positive.tsv",
        "manually-classified-tweets.tsv",
        "imdb-sentiment.tsv",
        "yelp-sentiment.tsv",
    ]
    df_from_each_file = (pd.read_csv(os.path.join("\\\\martin-server\\j\\Mute\\ml-datasets\\sentiment\\training", f), sep='\t', header=None, error_bad_lines=False) for f in filenames)
    d = pd.concat(df_from_each_file, ignore_index=True)
    d[0] = d[0].astype(str)
    d[1] = d[1].astype(float)
    return d

def main(tensorboard):
    data = load_data()
    print("Dataset: " + str(data.shape))

    max_features = 1000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data[0].values)
    print("Tokenizer fit complete")

    X = tokenizer.texts_to_sequences(data[0].values)
    X = pad_sequences(X, maxlen=64, padding="pre")
    Y = pd.get_dummies(data[1]).values
    print(Y)
    return

    embed_dim = 64
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

    batch_size = 128
    model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, verbose=2, validation_split = 0.25, callbacks=[
        tensorboard,
        keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    ])

    #model.fit_generator(buckets, epochs=10, callbacks=[
    #    tensorboard,
    #    keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    #])

    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))