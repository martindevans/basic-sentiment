import json

from pymagnitude import Magnitude

from gensim.models.keyedvectors import KeyedVectors

import pretrained_w2v.parameters as pr
import pretrained_w2v.model as mdl
import pretrained_w2v.dataset as dat

def main(tensorboard):

    print("## Building model")
    model = mdl.build()
    print(model.summary())

    print("## Loading Pretrained Word Vectors")
    word_vectors = KeyedVectors.load_word2vec_format(pr.pretrained_word_vectors_path, binary=True, limit=pr.vocab_size)

    print("## Loading Data")
    labelled_data = dat.load_sentiment_directory(pr.training_directory_path)

    print("## Building Training Dataset")
    dataset = dat.Dataset(labelled_data, word_vectors, pr.min_sentence_length, pr.max_sentence_length)
    print(json.dumps(dataset.stats(), sort_keys=True, indent=4, separators=(',', ': ')))

    print("## Training Model")
    mdl.train(model, dataset, tensorboard)