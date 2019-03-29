import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import text_to_word_sequence

import oov.parameters as pr

class WordPredictor:
    def __init__(self, model, pretrained_word_index):
        self.pretrained_word_index = pretrained_word_index
        self.model = model

    def predict(self, missing_word, sentence):
        # Split sentence string into words (with stops removed)
        seq = text_to_word_sequence(sentence, lower=True, split=" ")

        # Keep a list of word not in the dictionary
        unknown = []

        # Convert words into word vectors
        input = np.zeros(shape=(1, len(seq), pr.word_vector_dimension))
        for word_index, word in enumerate(seq):
            if word != missing_word and word in self.pretrained_word_index:
                input[0,word_index,:] = self.pretrained_word_index[word]
            else:
                if word not in unknown:
                    unknown.append(word)

        if len(unknown) == 0:
            raise Exception("Sentence has no unknown words")
        elif len(unknown) > 1:
            print("More than one unknown word: ")
            print(str(unknown))
            raise Exception("Sentence has multiplt unknown words")

        # Evaluate model
        output_vector = self.model.predict(input)

        # Find 10 similar words
        similar = []
        for item in self.pretrained_word_index.wv.similar_by_vector(output_vector[0], topn=10):
            similar.append(item)

        return (missing_word, similar, output_vector)

def build():
    model = Sequential()

    model.add(Dropout(pr.input_dropout, input_shape=(None,pr.word_vector_dimension)))

    model.add(LSTM(pr.lstm_output, dropout=pr.lstm_dropout, recurrent_dropout=pr.lstm_recurrent_dropout, return_sequences=True))
    model.add(LSTM(pr.lstm_output, dropout=pr.lstm_dropout, recurrent_dropout=pr.lstm_recurrent_dropout))

    ## Add some dropout before passing to the dense stack
    model.add(Dropout(pr.dense_pre_dropout))

    ## Add dense layers after the LSTM, shrinking down towards the output size
    size = int(pr.lstm_output * pr.dense_size_decay)
    while size > pr.word_vector_dimension:
        model.add(Dense(size, activation='tanh'))
        size = int(size * pr.dense_size_decay)

    ## Add a single layer of the output size
    model.add(Dense(pr.word_vector_dimension, activation='tanh'))

    print(" - Compiling model")
    model.compile(loss='cosine_proximity', optimizer='nadam', metrics = ['mse', 'mae', 'cosine_proximity'])

    return model

def epoch_callback(model, dataset):
    for _ in range(10):
        (sentence, (train_in, _), knockout_index) = dataset.single_random_val_item()
        to_print = list(sentence)
        to_print[knockout_index] = "<<" + to_print[knockout_index] + ">>"
        print(" ".join(to_print))

        predicted = model.predict(train_in)
        for item in dataset.pretrained_word_index.wv.similar_by_vector(predicted[0], topn=10):
            print(" - " + str(item))

def train(model, tensorboard, dataset):
    model.fit_generator(dataset.data_gen(), steps_per_epoch=dataset.steps_per_epoch * pr.training_reruns, epochs=pr.max_epochs, callbacks=[
        tensorboard,
        LambdaCallback(on_epoch_end=lambda batch, logs: epoch_callback(model, dataset)),
        #EarlyStopping(monitor='val_acc', patience=15, restore_best_weights=True),
        #ModelCheckpoint("models/sentiment-oov-weights.{epoch:02d}-{val_acc:.4f}.hdf5", monitor="val_acc", save_best_only=True),
    ], validation_data=dataset.val_gen(), validation_steps=dataset.val_steps * pr.validation_reruns)