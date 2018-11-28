import os
import json
import pathlib
import pickle
import gzip
import io
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import argparse as ap
import jsonpickle as jp

import w2v.model as mdl
import w2v.data as dat
import w2v.similarity as sim
import w2v.couples as cpl
import w2v.parameters as pr

from keras.preprocessing.sequence import make_sampling_table, skipgrams

def main(tensorboard):

    parser = ap.ArgumentParser(prog="W2V", description='Train a Word2Vec model')
    parser.add_argument("--rebuild-dataset", dest="rebuild", action='store_const', const=True, default=False, help='Force a complete rebuild of the dataset on disk')
    parser.add_argument("--skip-training", dest="skip_train", action='store_const', const=True, default=False, help='Do not train the model')
    args = parser.parse_args()

    set_path = pathlib.Path(os.path.join(pr.dataset_cache, "dataset.lzma"))
    cpl_path = pathlib.Path(os.path.join(pr.dataset_cache, "couples.lzma"))

    if args.rebuild:
        build_dataset(set_path, cpl_path)

    if not args.skip_train:
        train(tensorboard, set_path, cpl_path)

def build_dataset(set_path, cpl_path):

    ## Load list of sentences from datasets
    print("## Loading Data")
    sentences = []
    sentences.extend(dat.load_gutenberg_sentences(pr.gutenberg_path, pr.min_sentence_length, pr.total_books))
    sentences.extend(dat.load_wikipedia_sentences(pr.wikipedia_path, pr.min_sentence_length, pr.total_wiki_lines))

    ## Build word dataset
    print("## Building Dictionary")
    dataset = dat.Dataset(sentences, pr.vocab_size, pr.min_sentence_length)
    print(json.dumps(dataset.stats(), sort_keys=True, indent=4, separators=(',', ': ')))

    ## Build training couples
    print("## Extracting Training Couples")
    samples = cpl.Couples(dataset, pr.vocab_size, pr.window_size, pr.negative_samples)
    print(" - {0} couples".format(samples.get_couples_count()))

    ## Save it all to disk
    print("## Saving Dataset")
    with gzip.open(set_path, "wb") as set_file:
        with io.BufferedWriter(set_file) as writer:
            pickle.dump(dataset, writer)
    with gzip.open(cpl_path, 'wb') as cpl_file:
        with io.BufferedWriter(cpl_file) as writer:
            pickle.dump(samples, writer)

def train(tensorboard, set_path, cpl_path):

    print("## Loading Dataset")

    dataset = None
    with gzip.open(set_path, "rb") as set_file:
        with io.BufferedReader(set_file) as reader:
            dataset = pickle.load(reader)

    samples = None
    with gzip.open(cpl_path, 'rb') as cpl_file:
        with io.BufferedReader(cpl_file) as reader:
            samples = pickle.load(reader)

    print(json.dumps(dataset.stats(), sort_keys=True, indent=4, separators=(',', ': ')))
    print(" - {0} couples".format(samples.get_couples_count()))

    ## Create 2 models, one which we train and one we use to read validation data
    print("## Building model")
    model, val_model = mdl.build(1, pr.vector_dimension, pr.vocab_size)

    ## Create a random generator which selects validation data
    sim_cb = sim.SimilarityCallback(pr.vocab_size, pr.valid_size, dataset, pr.valid_window, val_model, 8)

    ## Train it
    print("Training model...")
    mdl.train(model, tensorboard, pr.epochs, pr.batch_size, samples, sim_cb, pr.negative_samples)

    ## Save for future use
    model.save("models/complete")