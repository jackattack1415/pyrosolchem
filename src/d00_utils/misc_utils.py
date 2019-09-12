import numpy as np

def normalize(unnormalized_array):
    """ Takes in an unnormalized array and normalizes it.

    Parameters
    ----------
    unnormalized_array : array
    array of some values that are unnormalized

    Returns
    -------
    normalized_array : array
    array of some values that are normalized
    """

    total = unnormalized_array.sum()
    normalized_array = np.true_divide(unnormalized_array, total)

    return normalized_array


def unpack_dictionary(dictionary):
    for key, value in dictionary.items():
        exec(key + '=val')

    return