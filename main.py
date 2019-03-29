import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np

import os
import time

def main():
    ## Only allocate GPU memory when needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess) 

    ## Save tensorboard logs into logs directory
    t = time.strftime("%a-%d-%b-%Y-%H-%M-%S", time.localtime())
    tensorboard = TensorBoard(log_dir="logs/{}".format(t))

    #from sentiment import sentiment
    #sentiment.main(tensorboard)

    #from w2v import main as w2vmain
    #w2vmain.main(tensorboard)

    #from pretrained_w2v import main as pretrainedmain
    #pretrainedmain.main(tensorboard)

    from oov import main as oovmain
    oovmain.main(tensorboard)

if __name__ == '__main__':
    main()