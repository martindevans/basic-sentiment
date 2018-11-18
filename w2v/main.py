import os
import json
import pathlib
import pickle

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

    if args.rebuild:
        build_dataset()

    if not args.skip_train:
        train(tensorboard)

def build_dataset():

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
    base_path = pr.dataset_cache
    set_path = pathlib.Path(os.path.join(base_path, "dataset.json"))
    cpl_path = pathlib.Path(os.path.join(base_path, "couples.json"))
    with set_path.open('wb') as set_file:
        pickle.dump(dataset, set_file)
    with cpl_path.open('wb') as cpl_file:
        pickle.dump(samples, cpl_file)

def train(tensorboard):

    print("## Loading Dataset")
    dataset = None
    with pathlib.Path(os.path.join(pr.dataset_cache, "dataset.json")).open('rb') as set_file:
        dataset = pickle.load(set_file)

    samples = None
    with pathlib.Path(os.path.join(pr.dataset_cache, "couples.json")).open('rb') as cpl_file:
        samples = pickle.load(cpl_file)

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