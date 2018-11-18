## Minimum length of sentence to accept
min_sentence_length = 6

## How many lines of the wikipedia dataset to load
total_wiki_lines = 500000

## How many books to load from the gutenberg dataset
total_books = 150

## How many valid words will there be?
vocab_size = 4500

## How far either side of the training word will we look?
window_size = 3

## What is the dimension of the embedding vector?
vector_dimension = 256

## How many negative samples will be generated compared to positive samples
negative_samples = 3

## How many training iterations will be run?
epochs = 250

## How many training items in a single epoch
batch_size = 1024

## How many words will we use for validation?
valid_size = 4

## We will only validate using words within the first NNN most common words
valid_window = 250

## Path to the gutenberg dammit dataset (https://github.com/aparrish/gutenberg-dammit)
gutenberg_path = "todo: path here"

## Path to wikipedia plain text dataset (https://sites.google.com/site/rmyeid/projects/polyglot)
wikipedia_path = "todo: path here"

## When the dataset is built where should it be stored?
dataset_cache = "data_cache"