import json
import numpy as np
from scipy.optimize import minimize

class WordEstimator:
    def __init__(self, word_vectors):
        self.current = []
        self.word_vectors = word_vectors

    # Extend estimator with a new vector
    def extend(self, result):
        self.current.append(result)

    # Return best guess word vector
    def best_guess(self):

        # geometric mean of an iterable of numbers
        def geo_mean(iterable):
            a = np.array(iterable)
            return a.prod() ** (1.0/len(a))

        # avg dot product of a given vector and all accumulated vectors
        def avg_dot(vec):
            return geo_mean(map(self.current, lambda v: np.dot(vec, v)))

        # Find the vector which minimises the geometric mean of the dot product of guess and each vector
        x0 = np.ones(300)
        result = minimize(avg_dot, x0, method='nelder-mead')

        return result.x