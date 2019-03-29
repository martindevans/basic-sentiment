import json
import os
import pickle
import gzip
import io
import pathlib

import argparse as ap

from gensim.models.keyedvectors import KeyedVectors

from keras.backend import set_learning_phase
from keras.models import load_model

import w2v.data as wvdat
import w2v.couples as wvcpl

import oov.parameters as pr
import oov.model as mdl
import oov.dataset as dat
import oov.accumulator as acc

def build_dataset(sent_path, word_vectors):

    ## Load list of sentences from datasets
    print("## Loading Data")
    sentences = []
    sentences.extend(wvdat.load_gutenberg_sentences(pr.gutenberg_path, pr.min_sentence_length, pr.total_books))
    sentences.extend(wvdat.load_wikipedia_sentences(pr.wikipedia_path, pr.min_sentence_length, pr.total_wiki_lines))

    ## Build dataset
    print("## Building Dictionary")
    dataset = dat.Dataset(sentences, word_vectors)
    print(json.dumps(dataset.stats(), sort_keys=True, indent=4, separators=(',', ': ')))

    ## Save it all to disk
    print("## Saving Dataset")
    with gzip.open(sent_path, "wb") as set_file:
        with io.BufferedWriter(set_file) as writer:
            pickle.dump(dataset, writer)

def train(tensorboard, sent_path, word_vectors):

    print("## Loading Dataset")
    dataset = None
    with gzip.open(sent_path, "rb") as set_file:
        with io.BufferedReader(set_file) as reader:
            dataset = pickle.load(reader)
    print(json.dumps(dataset.stats(), sort_keys=True, indent=4, separators=(',', ': ')))
    dataset.set_word_embedding(word_vectors)

    ## Create 2 models, one which we train and one we use to read validation data
    print("## Building model")
    model = mdl.build()
    print(model.summary())

    ## Train it
    print("Training model...")
    mdl.train(model, tensorboard, dataset)

    return model

def main(tensorboard):
    parser = ap.ArgumentParser(prog="OOV", description='Learn to estimate word vectors for new words')
    parser.add_argument("--rebuild-dataset", dest="rebuild", action='store_const', const=True, default=False, help='Force a complete rebuild of the dataset on disk')
    parser.add_argument("--skip-training", dest="skip_train", action='store_const', const=True, default=False, help='Do not train the model')
    args = parser.parse_args()

    print("## Loading Pretrained Word Vectors")
    word_vectors = KeyedVectors.load_word2vec_format(pr.pretrained_word_vectors_path, binary=True, limit=pr.vocab_size)

    sent_path = pathlib.Path(os.path.join(pr.dataset_cache, "sentences.lzma"))

    if args.rebuild:
        build_dataset(sent_path, word_vectors)

    model = None
    if not args.skip_train:
        model = train(tensorboard, sent_path, word_vectors)

        print("## Saving Model")
        set_learning_phase(0) # Disable training phase, this ensures all train-only layers are disabled from now on
        model.save("models/oov_final_model")
    else:
        print("## Loading Model")
        model_name = input("models/")
        model = load_model("models/" + model_name)

    predictor = mdl.WordPredictor(model, word_vectors)
    while True:
        word = input("What word are you trying to estimate? ")
        print("Now type example sentences using this word. Type `restart` to start a new word or `exit` to quit")
        accumulator = acc.WordEstimator(word_vectors);

        while True:
            text = input("Example > ")
            if text == "restart":
                break
            elif text == "exit":
                return;
            try:
                (missing_word, similar, output_vector) = predictor.predict(word, text)
                accumulator.extend(output_vector)
                print(json.dumps(similar))

                # Find 10 similar words
                v = accumulator.best_guess()
                similar = []
                for item in word_vectors.wv.similar_by_vector(output_vector[0], topn=10):
                    print(" - " + json.dumps(item));

            except Exception as e:
                print(e)