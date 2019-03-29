## Number of word vectors
vocab_size = 50000

## Path to pretrained embedding vectors
pretrained_word_vectors_path = "\\\\martin-server\\j\\Mute\\ml-models\\word2vec\\GoogleNews-vectors-negative300.bin.gz"

## Length of word vectors
word_vector_dimension = 300

## Path to the gutenberg dammit dataset (https://github.com/aparrish/gutenberg-dammit)
gutenberg_path = "\\\\martin-server\\j\\Mute\\ml-datasets\\language\\gutenberg-dammit-files-v002.zip"

## Path to wikipedia plain text dataset (https://sites.google.com/site/rmyeid/projects/polyglot)
wikipedia_path = "\\\\martin-server\\j\\Mute\\ml-datasets\\wikipedia\\full.txt"

## How many lines of the wikipedia dataset to load
total_wiki_lines = 10000000

## How many books to load from the gutenberg dataset
total_books = 450

## Minimum/Maximum length of sentence to accept
min_sentence_length = 7
max_sentence_length = 22

# Max number of items in a single batch
batch_max_sentences = 500

## What percentage of data becomes validation data
validation_chance = 0.05

## Maximum number of training epoches
max_epochs = 45

## When the dataset is built where should it be stored?
dataset_cache = "data_cache"

# Dropout used during training
input_dropout = 0
lstm_dropout = 0.1
lstm_recurrent_dropout = 0.1

# Dimension of LSTM output
lstm_output = int(word_vector_dimension * 1.1)

# Dropout before passing to the dense layer
dense_pre_dropout = 0.1

# How much to make each dense layer narrower until the correct output size is reached
dense_size_decay = 0.0

# How many times will the total training data be passed over per epoch
training_reruns = 3

# How many times will the total validation data be passed over per epoch
validation_reruns = 3