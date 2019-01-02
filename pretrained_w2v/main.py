import json
import time

from gensim.models.keyedvectors import KeyedVectors

from keras.models import load_model

import pretrained_w2v.parameters as pr
import pretrained_w2v.model as mdl
import pretrained_w2v.dataset as dat
import pretrained_w2v.evaluate as eva

def main(tensorboard):

    # todo: It should be possible to load a saved hdf5 file and continue training
    # - How should the dataset be handled? Seed the train/validation split so it's consistent?

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

    print("## Saving Model")
    name = "final_model"
    model.save("models/" + name)

    print("## Evaluating Model")
    predictor = mdl.SentimentAnalyser(model, word_vectors)
    (report, conf_mat) = eva.evaluate(dataset, predictor)
    print(report)
    print(conf_mat)

    while True:
        text = input("Analyse > ")
        if text == "exit":
            break
        print(list(predictor.sentiment(text)[0]))