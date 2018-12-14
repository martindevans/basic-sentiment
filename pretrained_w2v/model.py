import numpy as np

import tensorflow as tf

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, GaussianNoise
from keras.models import Sequential
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from nltk.corpus import stopwords

from utils.tensorflow_metrics import as_keras_metric
import pretrained_w2v.parameters as pr

class SentimentAnalyser:
    def __init__(self, model, word_vectors):
        self.model = model
        self.pretrained_word_index = word_vectors
        self.stops = set(stopwords.words('english'))

    def sentiment(self, sentence):
        # Split sentence string into words (with stops removed)
        seq = text_to_word_sequence(sentence, lower=True, split=" ")
        seq = list([w for w in seq if not w in self.stops])

        # Convert words into word vectors
        input = np.zeros(shape=(1, len(seq), pr.word_vector_dimension))
        for word_index, word in enumerate(seq):
            if word in self.pretrained_word_index:
                input[0,word_index,:] = self.pretrained_word_index[word]

        # Evaluate model
        output = self.model.predict(input)

        return output

def build():
    model = Sequential()

    model.add(GaussianNoise(pr.input_std_dev, input_shape=(None,pr.word_vector_dimension)))

    ## LSTM layer works through all the words of the sentence
    model.add(Bidirectional(LSTM(pr.lstm_output, dropout=pr.dropout, recurrent_dropout=pr.recurrent_dropout)))

    ## Add dense layers after the LSTM
    if (pr.intermediate_dense_size > 0):
        for _ in range(0, pr.intermediate_dense_layers):
            model.add(Dense(pr.intermediate_dense_size, activation='selu'))

    ## 3 neuron dense layer classifies as  one of the three classes (Positive, Neutral, Negative)
    model.add(Dense(3, activation='softmax'))

    print(" - Compiling model")
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics = ['accuracy'])

    return model

def train(model, dataset, tensorboard):
    model.fit_generator(dataset.data_gen(), steps_per_epoch=dataset.steps_per_epoch, epochs=pr.max_epochs, callbacks=[
        tensorboard,
        EarlyStopping(monitor='val_acc', patience=8, restore_best_weights=True),
        ModelCheckpoint("models/sentiment-w2v-weights.{epoch:02d}-{val_acc:.4f}.hdf5", monitor="val_acc", save_best_only=True),
    ], validation_data=dataset.val_gen(), validation_steps=dataset.val_steps)