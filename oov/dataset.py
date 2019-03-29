import random
import uuid

import numpy as np

from keras.preprocessing.text import text_to_word_sequence

import oov.parameters as pr

### Given a large set of sentences create sentences with a single word removed, paired with the vector of the removed word
class Dataset():
    def __init__(self, sentences, pretrained_word_index):

        def check_sentence(words):
            for word in words:
                if word not in pretrained_word_index:
                    return False
            return True

        def add_to_batch(lookup, index, item, maxn):
            li = lookup.setdefault(index, [])
            if len(li) >= maxn:
                lookup[uuid.uuid4()] = li
                lookup[index] = [sentence]
            else:
                li.append(item)

        batches_lookup = {}
        val_lookup = {}
        kept = 0
        discarded = 0
        bad_length = 0
        non_embedded = 0
        for sentence in sentences:
            # Convert sentence to words
            l = len(sentence)
            if l < pr.min_sentence_length or l > pr.max_sentence_length:
                discarded += 1
                bad_length += 1
                continue
            # Eliminate sentences which cannot be completely converted to word vectors
            if not check_sentence(sentence):
                discarded += 1
                non_embedded += 1
                continue
            # Store list of words in a batch of other sentences with the same length
            kept += 1

            if random.random() < pr.validation_chance:
                add_to_batch(val_lookup, l, sentence, pr.batch_max_sentences)
            else:
                add_to_batch(batches_lookup, l, sentence, pr.batch_max_sentences)

        self.val_batches = list(val_lookup.values())
        self.train_batches = list(batches_lookup.values())
        self.total_sentences = kept
        self.discarded = discarded
        self.discarded_bad_length = bad_length
        self.discarded_bad_embedding = non_embedded

        # Run each batch once per epoch
        self.steps_per_epoch = len(self.train_batches)
        self.val_steps = len(self.val_batches)

    def set_word_embedding(self, word_embeddings):
        self.pretrained_word_index = word_embeddings

    def stats(self):
        return {
            "train_batches": list(map(lambda b: len(b), self.train_batches)),
            "val_batches": list(map(lambda b: len(b), self.val_batches)),
            "discarded_sentences": self.discarded,
            "discarded_bad_length": self.discarded_bad_length,
            "discarded_bad_embedding": self.discarded_bad_embedding,
            "total_sentences": self.total_sentences,
            "steps_per_epoch": self.steps_per_epoch,
            "validation_steps": self.val_steps
        }

    def choose_knockout(self, sentence):
        return random.randint(0, len(sentence) - 1)

    def convert_batch(self, batch, knockout_index = None):
        num_sentences = len(batch)
        len_sentence = len(batch[0])
        size_word = pr.word_vector_dimension

        train_input = np.zeros(shape=(num_sentences, len_sentence, size_word))
        train_output = np.zeros(shape=(num_sentences, size_word))

        for sentence_index, sentence in enumerate(batch):
            dead_word_index = knockout_index or self.choose_knockout(sentence)
            for word_index, word in enumerate(sentence):
                if word_index == dead_word_index:
                    train_output[sentence_index,:] = self.pretrained_word_index[word]
                else:
                    train_input[sentence_index,word_index,:] = self.pretrained_word_index[word]

        return (train_input, train_output)

    def single_random_val_item(self):
        result = [ random.choice(random.choice(self.val_batches)) ]
        knockout = random.randint(0, len(result[0]) - 1)
        return (list(result[0]), self.convert_batch(result, knockout), knockout)

    def data_gen(self):
        batch_num = 0
        while True:
            print(batch_num)
            batch_num += 1
            
            random.shuffle(self.train_batches)
            for batch in self.train_batches:
                yield self.convert_batch(batch)

    def val_gen(self):
        while True:
            random.shuffle(self.val_batches)
            for batch in self.val_batches:
                yield self.convert_batch(batch)