import numpy as np

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, GaussianNoise
from keras.models import Sequential
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint

import pretrained_w2v.parameters as pr

def build():
    model = Sequential()

    model.add(GaussianNoise(pr.input_std_dev, input_shape=(None,pr.word_vector_dimension)))

    ## LSTM layer works through all the words of the sentence
    lstm = LSTM(pr.lstm_output, dropout=pr.dropout, recurrent_dropout=pr.recurrent_dropout)
    if pr.bidirectional:
        model.add(Bidirectional(lstm))
    else:
        model.add(lstm)

    ## A single dense layer modifies the LSTM output
    if (pr.intermediate_dense_size > 0):
        model.add(Dense(pr.intermediate_dense_size, activation='tanh'))

    ## 3 neuron dense layer classifies as  one of the three classes (Positive, Neutral, Negative)
    model.add(Dense(3, activation='softmax'))

    print(" - Compiling model")
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics = ['accuracy'])

    return model

def train(model, dataset, tensorboard):
    model.fit_generator(dataset.data_gen(), steps_per_epoch=dataset.steps_per_epoch, epochs=pr.max_epochs, callbacks=[
        tensorboard,
        EarlyStopping(monitor='val_acc', patience=8, restore_best_weights=True),
        ModelCheckpoint("models/sentiment-w2v-weights.{epoch:02d}-{loss:.2f}.hdf5", monitor="val_acc", save_best_only=True),
    ], validation_data=dataset.val_gen(), validation_steps=dataset.val_steps)