# Training parameters
## LSTM dropout
dropout = 0.2
## LSTM recurrent dropout
recurrent_dropout = 0.2
## Maximum number of training epochs
max_epochs = 100
## Max number of sentences in a single batch
max_batch_size = 400
## How many validation batches pert raining batch
validation_pct = 0.2

# Dataset parameters
## Number of words to load
vocab_size = 25000
## Whether or not a pretrained embedding should be used
use_pretrained_embedding = True
## Path to pretrained embedding vectors (only if pretrained embedding vectors are used)
pretrained_word_vectors_path = "\\\\martin-server\\j\\Mute\\ml-models\\word2vec\\GoogleNews-vectors-negative300.bin.gz"
## Dimension of the embedding vectors (only if pretrained embedding vectors are not used)
embed_dim = 300
## Minimum length of sentence to include in the dataset
min_sentence_length = 4
## Maximum length of sentence to include in the dataset
max_sentence_length = 512
## Path to sentiment training data directory
training_directory_path = "\\\\martin-server\\j\\Mute\\ml-datasets\\sentiment\\training"

# Model parameters
## Whether the LSTM layer should be bidirectional
bidirectional = True
## Dimension of the LSTM output
lstm_output = 128