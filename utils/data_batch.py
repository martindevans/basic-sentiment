from keras import utils

## Bucket a sequence of data up by length
class SequenceBucketLength(utils.Sequence):

    def __init__(self, x, y):
        self.foo = 1;