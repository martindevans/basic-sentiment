import os
import math
import collections
import pathlib
import re

import pandas as pd
import numpy as np
import nltk as nltk
import progressbar as pb

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from utils.gutenbergdammit.ziputils import search, searchandretrieve, retrieve_one, loadmetadata

class Dataset:

    def __init__(self, sentence_lists, max_features, min_sequence_length):

        sentence_lists = [sentence for sentence in sentence_lists if len(sentence) >= min_sequence_length]

        ## Form list of all words
        words = list([val for sublist in sentence_lists for val in sublist])

        ## Count occurences of each word
        count = [('UNK', -1)]
        count.extend(collections.Counter(words).most_common(max_features))

        ## Assign each word an ID, with lower indices being more frequently occuring words
        dictionary = dict()
        for word, _ in count:
            id = len(dictionary)
            dictionary[word] = id

        ## Convert sentences (lists of strings) into sentences (list of word IDs)
        data = []
        unk_count = 0
        for sentence in sentence_lists:
            converted = []
            for word in sentence:
                if word in dictionary:
                    converted.append(dictionary[word])
                else:
                    unk_count += 1
                    converted.append(0)
            data.append(converted)

        ## Form decode dictionary (ID -> word)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        ## Save all the data
        self.data = data
        self.count = count
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary


    def stats(self):
        return {
            "num_words": len(self.count),
            "num_sentences": len(self.data),
        }

    ## Get a list of tuples of word & count of occurences of that word
    def get_counts(self):
        return self.count

    ## Get a list of lists of word IDs (sentences)
    def get_data(self):
        return self.data

    ## Get a map from word -> ID
    def get_lookup(self):
        return self.dictionary

    ## Get a map from ID -> word
    def get_reverse_lookup(self):
        return self.reverse_dictionary

## Take a string (an entire book) and split into a List<List<String>>
def __sentences_from_str(book, min_sentence_length):
    #Split book into list of sentences
    sentences = nltk.sent_tokenize(book)

    #Split sentences into lists of words
    result = []
    for sentence in sentences:
        clean = re.sub("'\"", "", sentence)
        clean = re.sub("[^a-zA-Z0-9]"," ", clean)
        words = text_to_word_sequence(clean, lower=True, split=" ")
        if len(words) >= min_sentence_length:
            result.append(words)

    return result

def load_wikipedia_sentences(path, min_sentence_length=6, lines=10000):

    with pb.ProgressBar(widgets=[ pb.Percentage(), ' ', pb.AdaptiveETA(), ' ', pb.Bar() ], max_value=lines) as bar:
        p = pathlib.Path(path)
        sentences = []
        with p.open(encoding="utf8") as f:
            for i in range(0, lines):
                sentences.extend(__sentences_from_str(f.readline(), min_sentence_length))
                bar.update(i)

        print(" - Wikipedia sentences: " + str(len(sentences)))
        return sentences

def load_gutenberg_sentences(path, min_sentence_length=6, total_books=100):
    ## Load books and split up into a List<List<String>>. Each inner list is a sentence (split into words)
    p = pathlib.Path(path)
    m = loadmetadata(p)
    sentence_list = []
    for info in search(m, {'Language': 'English'}):

        # Skip this book if we don't know when the author was born
        if not "Author Birth" in info or len(info["Author Birth"]) == 0: continue

        # Skip book if birthday cannot be parsed as int
        birthday = info["Author Birth"][0]
        try:
            birthday = int(birthday)
        except:
            continue

        # Skip this book if it's too old
        if birthday < 1925: continue

        text = retrieve_one(p, info['gd-path'])
        print(info['Title'][0], info['Num'], len(text), birthday)
        sentence_list.extend(__sentences_from_str(text, min_sentence_length))

        total_books -= 1
        if (total_books <= 0):
            break

    print(" - Gutenberg sentences: " + str(len(sentence_list)))
    return sentence_list