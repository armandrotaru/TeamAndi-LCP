import numpy as np



def search_sequence_numpy(arr, seq):
    """Find sequence positions within array.

    Parameters
    ----------
    arr : int array, shape (n_tokens_arr)
        Token array, where n_tokens_arr is the number of tokens in the array.

    seq : int array, shape (n_tokens_seq)
        oken sequence to be located within the array, where n_tokens_seq is the number of tokens in the
        sequence.

    Returns
    -------
    int array, shape (n_pos)
        Positions occupied by the sequence in the array, where n_pos is the number of positions.

    Adapted from https://stackoverflow.com/a/36535397
    """

    # store sizes of input array and sequence
    n_tokens_arr, n_tokens_seq = arr.size, seq.size

    # compute range of sequence
    r_seq = np.arange(n_tokens_seq)

    # create a 2D array of sliding indices across the entire length of input array, then match up with the
    # input sequence and get the matching starting indices
    M = (arr[np.arange(n_tokens_arr - n_tokens_seq + 1)[:,None] + r_seq] == seq).all(1)

    # get the range of those indices as final output
    if M.any() > 0:

        return np.where(np.convolve(M, np.ones((n_tokens_seq), dtype=int)) > 0)[0]

    else:

        return []


