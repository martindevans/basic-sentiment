from multiprocessing import Pool
import os

import numpy as np

from keras.preprocessing.sequence import make_sampling_table, skipgrams

class CoupleFunc(object):
    def __init__(self, vocab_size, window_size, sampling_table, negative_samples):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.sampling_table = sampling_table
        self.negative_samples = negative_samples

    def __call__(self, sentence):
        c, l = skipgrams(sentence, self.vocab_size, window_size=self.window_size, sampling_table=self.sampling_table, negative_samples=self.negative_samples)
        return (c, l)

class Couples:

    def __init__(self, dataset, vocab_size, window_size, negative_samples):
        ## Create dataset of pairs of words. Each pair is either a pair of words which appeared within the same window (with a correspoding positive label)
        ## or a pair of words which never appeared together (with a correspoding negative label)
        sampling_table = make_sampling_table(vocab_size + 1)
        couples = []
        labels = []

        with Pool(os.cpu_count()) as pool:
            para_results = pool.imap_unordered(CoupleFunc(vocab_size, window_size, sampling_table, negative_samples), dataset.get_data(), chunksize=512)
            for result in para_results:
                (c, l) = result
                couples.extend(c)
                labels.extend(l)

        ## Create array of target word and context word
        word_target, word_context = zip(*couples)
        self.word_target = np.array(word_target, dtype="int32")
        self.word_context = np.array(word_context, dtype="int32")
        self.labels = labels

    def get_couples_count(self):
        return len(self.word_target)

    def get_couple_data(self, index):
        return (self.word_target[index], self.word_context[index], self.labels[index])