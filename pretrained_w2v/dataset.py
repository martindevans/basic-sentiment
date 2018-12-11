import os
import math
import random
import time

from os.path import join
from multiprocessing import Pool

import pandas as pd
import numpy as np
import progressbar as pb

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import pretrained_w2v.parameters as pr

class Dataset:

    def __init__(self, labelled_items, pretrained_word_index, min_sequence_length, max_sequence_length):

        self.pretrained_word_index = pretrained_word_index

        # Split labelled data into a list of sentences and a list of classes, making sure to discard invalid items
        (sentences, classes) = labelled_items
        if (len(sentences) != len(classes)):
            raise "Data and class labels are not the same length"

        # Zip together into a list of [(sentence, class)]
        seq_class = list(zip(sentences, classes))

        # Store sequences in the smallest batch which can contain them
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

        def chunks(l, maxn):
            results = []

            current = None
            count = 0
            for item in l:
                (words,clss) = item
                c = len(words)
                if current != None and count + c < maxn:
                    current.append((words,clss))
                    count = count + c
                else:
                    current = []
                    results.append(current)
                    count = 0

            return results


        def pad_to_max(lists, pad_with):
            max_len = max(map(len, lists))
            for item in lists:
                l = len(item)
                if l < max_len:
                    item += [pad_with] * (max_len - l)

        # Each batch is:
        #   `List<Tuple<List<str>, Classification>>`
        # But we want to split that into
        #   `Tuple<List<List<str>>, List<Classification>>`
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

                # Pad data to longest list in set
                pad_to_max(data, "<unk>")

                # Append to the output list
                output.append((data, clss))

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
                "example": str(sentences[0][:5]),
                "example_class": str(classes[0])
            }

        train_batch_stats = list(map(batch_stats_map, self.train_batches))
        val_batch_stats = list(map(batch_stats_map, self.validation_batches))

        return {
            #"train_batches": sorted(train_batch_stats, key=lambda x: x["num_items"]),
            #"validation_batches": sorted(val_batch_stats, key=lambda x: x["num_items"]),
            "num_train_batches": len(self.train_batches),
            "num_validation_batches": len(self.validation_batches),
            "num_sentences": self.total_sequences,
            "longest_sentence": self.max_sequence_length
        }

    def convert_batch(self, batch):
            (data, classes) = batch
            result = np.zeros(shape=(len(data), len(data[0]), pr.word_vector_dimension))
            for sentence_index, sentence in enumerate(data):
                for word_index, word in enumerate(sentence):
                    if word in self.pretrained_word_index:
                        result[sentence_index,word_index,:] = self.pretrained_word_index[word]
            return (result, classes)

    def data_gen(self):
        while True:
            random.shuffle(self.train_batches)
            for batch in self.train_batches:
                yield self.convert_batch(batch)

    def val_gen(self):
        while True:
            random.shuffle(self.validation_batches)
            for batch in self.validation_batches:
                yield self.convert_batch(batch)

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