import math

def batch_by_len(items, min_sequence_length, max_sequence_length, len_of_item, validate_item):
    batches_lookup = {}
    max_sequence_length = 0
    total_sequences = 0
    for item in items:

        ## Get the length of the sentence and discard it if necessary
        l = len_of_item(item)
        if l == 0 or l < min_sequence_length or l > max_sequence_length:
            continue

        ## Apply additional validation to item
        if validate_item and not validate_item(item):
            continue

        max_sequence_length = max(max_sequence_length, l)
        total_sequences += 1

        ## Get the batch with the correct size (pow2 of batch index is > sentence length)
        base_pow = math.ceil(math.log2(l))
        batches_lookup.setdefault(base_pow, []).append(item)

    return batches_lookup, max_sequence_length, total_sequences
