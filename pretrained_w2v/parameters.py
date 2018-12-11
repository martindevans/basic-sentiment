# Training parameters
## LSTM dropout
dropout = 0.2
## LSTM recurrent dropout
recurrent_dropout = 0.2
## Std dev of noise applied to input
input_std_dev = 0.0
## Maximum number of training epochs
max_epochs = 100
## Max number of words in a single batch (tweak this up as much as possible until you run out of GPU memory)
max_batch_size = 100000

# Dataset parameters
## Number of word vectors
vocab_size = 500000
## Path to pretrained embedding vectors
pretrained_word_vectors_path = "\\\\martin-server\\j\\Mute\\ml-models\\word2vec\\GoogleNews-vectors-negative300.bin.gz"
## Length of word vectors
word_vector_dimension = 300
## Minimum length of sentence to include in the dataset
min_sentence_length = 4
## Maximum length of sentence to include in the dataset
max_sentence_length = 512
## Path to sentiment training data directory
training_directory_path = "\\\\martin-server\\j\\Mute\\ml-datasets\\sentiment\\training"
## What percentage of the input data is validation data
validation_pct = 0.2

# Model parameters
## Dimension of the LSTM output
lstm_output = 64
## Number of neurons per layer after the LSTM
intermediate_dense_size = lstm_output
## Number of layers after the LSTM
intermediate_dense_layers = 2