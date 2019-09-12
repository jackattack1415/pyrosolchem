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

    total = unnormalized_array.sum
    normalized_array = unnormalized_array / total

    return normalized_array