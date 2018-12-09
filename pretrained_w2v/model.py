import numpy as np

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.models import Sequential
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint

import pretrained_w2v.parameters as pr

def build(word_vectors = None):
    model = Sequential()

    ## Add an untrained embedding layer if a pretrained set of word vectors was no supplied
    if word_vectors == None:
        model.add(Embedding(pr.vocab_size, pr.embed_dim, trainable=True))
    else:
        model.add(word_vectors.get_keras_embedding(train_embeddings=False))

    ## LSTM layer works through all the words of the sentence
    lstm = LSTM(pr.lstm_output, dropout=pr.dropout, recurrent_dropout=pr.recurrent_dropout)
    if pr.bidirectional:
        model.add(Bidirectional(lstm))
    else:
        model.add(lstm)

    ## 3 neuron dense layer classifies as  one of the three classes (Positive, Neutral, Negative)
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics = ['accuracy'])
    return model

def train(model, dataset, tensorboard):
    model.fit_generator(dataset.data_gen(), steps_per_epoch=dataset.steps_per_epoch, epochs=pr.max_epochs, callbacks=[
        tensorboard,
        EarlyStopping(monitor='val_acc', patience=4, restore_best_weights=True),
        ModelCheckpoint("models/sentiment-w2v-weights.{epoch:02d}-{loss:.2f}.hdf5", monitor="val_acc", save_best_only=True),
    ], validation_data=dataset.val_gen(), validation_steps=dataset.val_steps)