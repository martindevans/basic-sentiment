import os
import math
import random

from os.path import join

import pandas as pd
import numpy as np
import progressbar as pb

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import pretrained_w2v.parameters as pr

class Dataset:

    def __init__(self, labelled_items, pretrained_word_index, vocab_size, min_sequence_length, max_sequence_length):

        self.unknown_words = 0
        self.known_words = 0

        # Split labelled data into a list of sentences and a list of classes, making sure to discard invalid items
        (sentences, classes) = labelled_items
        if (len(sentences) != len(classes)):
            raise "Data and class labels are not the same length"

        sequences = None
        if pretrained_word_index == None:
            # Fit a tokenizer, converting words to integers
            self.tokenizer = Tokenizer(num_words=vocab_size)
            self.tokenizer.fit_on_texts(sentences)
            sequences = self.tokenizer.texts_to_sequences(sentences)
        else:
            # Convert words into indices from pretrained model
            sequences = []
            with pb.ProgressBar(widgets=[ pb.Percentage(), ' ', pb.AdaptiveETA(), ' ', pb.Bar() ], max_value=len(sentences)) as bar:
                i = 0
                for item in sentences:
                    bar.update(i)
                    i += 1
                    seq = []
                    for word in item:
                        windex = pretrained_word_index.wv.vocab.get(word)
                        if windex:
                            seq.append(windex.index)
                            self.known_words += 1
                        else:
                            seq.append(0)
                            self.unknown_words += 1
                    sequences.append(seq)

        ## Zip together sentences with classes to form tuples of each sentence with it's class
        seq_class = list(zip(sequences, classes))

        # Store sequences in the smallest batch which can contain them
        # Batches have a pow2 size and items are padded up to the appropriate length
        batches_lookup = {}
        self.max_sequence_length = 0
        self.total_sequences = 0
        for tup in seq_class:

            ## Get the length of the sentence and discard it if necessary
            l = len(tup[0])
            if l == 0 or l < min_sequence_length or l > max_sequence_length:
                continue

            ## Check that the class label is valid
            if tup[1] < 0 or tup[1] > 3 or math.isnan(tup[1]):
                continue

            self.max_sequence_length = max(self.max_sequence_length, l)
            self.total_sequences += 1

            ## Get the batch with the correct size (pow2 of batch index is > sentence length)
            base_pow = math.ceil(math.log2(l))
            batches_lookup.setdefault(base_pow, []).append(tup)

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        # Each batch is:
        #   `List<Tuple<List<Int>, Classification>>`
        # But we want to split that into
        #   `Tuple<List<List<Int>>, List<Classification>>`
        # This can be achieved by magic: https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
        self.train_batches = []
        self.validation_batches = []
        for key in batches_lookup:
            value = batches_lookup[key]

            # Split into chunks limited by the batch size
            chunked = chunks(value, pr.max_batch_size)

            for ck in chunked:
                # Convert list of tuples [(data, class)] into tuple of lists ([data], [class])
                (data, clss) = np.array(list(map(list, zip(*ck))))

                # Convert classes (0, 1, 2) to one hot vectors
                clss = to_categorical(clss, 3)

                # Output batch as either training or validation batch
                output = self.train_batches
                if (random.random() < pr.validation_pct):
                    output = self.validation_batches
                output.append((pad_sequences(data), clss))

        ## Shuffle batch order
        random.shuffle(self.train_batches)
        random.shuffle(self.validation_batches)

        # Run each batch once per epoch
        self.steps_per_epoch = len(self.train_batches)
        self.val_steps = len(self.validation_batches)

    def stats(self):

        def batch_stats_map(batch):
            sentences = batch[0]
            classes = batch[1]

            return {
                "num_items": len(sentences),
                "item_length": len(sentences[0]),
                #"example": str(sentences[0][:32]),
                #"example_class": str(classes[0])
            }

        train_batch_stats = list(map(batch_stats_map, self.train_batches))
        val_batch_stats = list(map(batch_stats_map, self.validation_batches))

        return {
            #"train_batches": sorted(train_batch_stats, key=lambda x: x["num_items"]),
            #"validation_batches": sorted(val_batch_stats, key=lambda x: x["num_items"]),
            "known_words": self.known_words,
            "unknown_words": self.unknown_words,
            "num_train_batches": len(self.train_batches),
            "num_validation_batches": len(self.validation_batches),
            "num_sentences": self.total_sequences,
            "longest_sentence": self.max_sequence_length
        }

    def data_gen(self):
        while True:
            random.shuffle(self.train_batches)
            for batch in self.train_batches:
                yield batch

    def val_gen(self):
        while True:
            random.shuffle(self.validation_batches)
            for batch in self.validation_batches:
                yield batch

    def get_train_batches(self):
        return self.train_batches

    def get_validation_batches(self):
        return self.validation_batches

def clean_sentences(sentences):
    results = []

    stops = set(stopwords.words('english'))
    with pb.ProgressBar(widgets=[ pb.Percentage(), ' ', pb.AdaptiveETA(), ' ', pb.Bar() ], max_value=len(sentences)) as bar:
        i = 0
        for s in sentences:
            bar.update(i)
            i += 1
            seq = text_to_word_sequence(s, lower=True, split=" ")
            seq = list([w for w in seq if not w in stops])
            results.append(seq)

    return results

def load_sentiment_directory(directory_path):
    ## Load a dataset from each file in the directory
    datasets = []
    for f in os.listdir(directory_path):
        ff = os.path.join(directory_path, f)
        if os.path.isfile(ff):
            print(" - " + str(ff))
            datasets.append(pd.read_csv(ff, sep='\t', header=None, error_bad_lines=False))

    ## Concat into one big dataset of strings
    d = pd.concat(datasets, ignore_index=True)

    ## Extract data and labels
    strings = list(d[0].astype(str).values)
    classes = list(d[1].astype(float).values)

    return (clean_sentences(strings), classes)