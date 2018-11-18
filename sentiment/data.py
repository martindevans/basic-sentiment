import os
import math

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Dataset:

    def __init__(self, items, max_features, min_sequence_length):

        # Fit a tokenizer, converting words to integers
        self.tokenizer = Tokenizer(num_words=max_features)
        self.tokenizer.fit_on_texts(items)
        sequences = self.tokenizer.texts_to_sequences(items)
        self.total_sequences = len(sequences)

        # Store sequences in the smallest batch which can contain them
        # Batches have a pow2 size and items are padded up to the appropriate length
        self.batches = {}
        self.max_sequence_length = 0
        for seq in sequences:
            l = len(seq)
            if l == 0 or l < min_sequence_length:
                continue
            self.max_sequence_length = max(self.max_sequence_length, l)

            base_pow = math.ceil(math.log2(l))
            self.batches.setdefault(base_pow, []).append(seq)

        # Pad each batch out to the length of the longest item in the batch
        for values in self.batches.values():
            pad_sequences(values)

    def stats(self):
        batch_stats = list(map(lambda b: {
            "id": b[0],
            "num_items": len(b[1]),
            "item_length": len(b[1][0])
        }, self.batches.items()))

        return {
            "num_batches": len(self.batches),
            "num_sentences": self.total_sequences,
            "longest_sentence": self.max_sequence_length,
            "batches": sorted(batch_stats, key=lambda x: x["id"])
        }

    def get_batches(self):
        return self.batches.values()

def load(words=1000, min_sentence_length=4):

    ## Read all the files
    filenames = [
        "neutral.txt",
        "acllmdb-negative.txt",
        "acllmdb-positive.txt",
        "manually-classified-tweets.tsv",
        "imdb-sentiment.txt",
        "yelp-sentiment.txt",
    ]
    df_from_each_file = (pd.read_csv(os.path.join("\\\\martin-server\\j\\Mute\\ml-datasets\\sentiment\\training", f), sep='\t', header=None, error_bad_lines=False, usecols=[0]) for f in filenames)

    ## Concat into one set
    d = pd.concat(df_from_each_file, ignore_index=True)

    ## Ensure it's interpreted as a string
    d = d[0].astype(str).values

    return Dataset(d, words, min_sentence_length);