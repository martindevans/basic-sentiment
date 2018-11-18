import numpy as np

class SimilarityCallback:

    ## Construct a new similarity callback
    ## - vocab_size: number of words in vocab
    ## - valid_size: Number of words to check
    ## - dataset: The complete dataset
    ## - valid_examples: A random generator which picks a random words in the dictionary
    ## - validation_model: The validation model output, which returns the similarity of two word vectors
    ## - top_k: How many most similar words to list (in order of similarity)
    def __init__(self, vocab_size, valid_size, dataset, valid_window, validation_model, top_k = 8):
        self.vocab_size = vocab_size
        self.valid_size = valid_size
        self.dataset = dataset
        self.valid_window = valid_window
        self.validation_model = validation_model
        self.top_k = top_k

    ## Print the N most similar words to a set of M validation words
    ## - N was top_k in the constructor
    ## - M was valid_size in the constructor
    def run_sim(self):
        rev_dic = self.dataset.get_reverse_lookup()
        top_k = self.top_k
        for i in range(self.valid_size):
            ex = np.random.randint(0, self.valid_window - 1)
            valid_word = rev_dic[ex]
            sim = self._get_sim(ex)
            nearest = (-sim).argsort()[1:top_k + 1]
            print("Nearest to " + valid_word + ":")
            for i in nearest:
                close_word = rev_dic[i]
                print(" - " + close_word)

    ## Get the similarity between `valid_word_idx` and every other word in the dictonary
    def _get_sim(self, valid_word_idx):
        sim = np.zeros((self.vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        vm = self.validation_model
        for i in range(self.vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            sim[i] = vm.predict_on_batch([in_arr1, in_arr2])
        return sim