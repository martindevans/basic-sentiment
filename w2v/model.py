import tensorflow as tf
import numpy as np
import math

from keras.initializers import TruncatedNormal
from keras.layers import Input, Dense, GaussianNoise
from keras.layers.core import Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Dot
from keras.optimizers import TFOptimizer
from keras.models import Model
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint

def build(seed, vector_dim, vocab_size):

    # Create input layer of the network
    input_target = Input((1,))
    input_context = Input((1,))
    embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

    target = Reshape((vector_dim, 1))(embedding(input_target))
    context = Reshape((vector_dim, 1))(embedding(input_context))

    # setup a cosine similarity operation which will be output in a secondary model
    similarity = Dot(0, True)([target, context])

    # now perform the dot product operation to get a similarity measure
    dot_product = Dot(1, False)([target, context])
    dot_product = Reshape((1,))(dot_product)

    # Insert some noise to prevent overfitting
    noise = GaussianNoise(0.00005)(dot_product)

    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(noise)

    # create the primary training model
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer="adam")

    # create a secondary validation model to run our similarity checks during training
    validation_model = Model(inputs=[input_target, input_context], outputs=similarity)

    return model, validation_model

def train(model, tensorboard, epochs, batch_size, couples, sim_cb, negative_samples):

    # Lower weight for negative classes if there are more of them
    class_weight = {
        0: min(1.0, 1.0 / negative_samples),
        1: 1.0,
    }

    total_couple_count = couples.get_couples_count()

    def gen():
        while True:
            # Copy a single piece of training data into arrays
            arr_1 = np.zeros((batch_size,))
            arr_2 = np.zeros((batch_size,))
            arr_3 = np.zeros((batch_size,))
            for mb in range(batch_size):
                idx = np.random.randint(0, total_couple_count - 1)
                (arr_1[mb], arr_2[mb], arr_3[mb]) = couples.get_couple_data(idx)
            yield ([arr_1, arr_2], arr_3)

    steps_per_epoch = total_couple_count / batch_size
    val_steps_per_epoch = steps_per_epoch * 0.04;

    model.fit_generator(gen(), steps_per_epoch, epochs, class_weight=class_weight, callbacks=[
        LambdaCallback(on_epoch_end=lambda batch,logs: sim_cb.run_sim()),
        EarlyStopping(monitor="loss", patience=3, min_delta=0.0001, restore_best_weights=True),
        ModelCheckpoint("models/w2v-weights.{epoch:02d}-{loss:.2f}.hdf5", monitor="loss", save_best_only=True),
        tensorboard
    ], validation_data=gen(), validation_steps=val_steps_per_epoch)

    sim_cb.run_sim()