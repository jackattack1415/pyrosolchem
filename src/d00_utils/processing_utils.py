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


def get_bootstrap_sample(dataset):
    bootstrap_sample = np.random.choice(dataset, size=len(dataset))

    return bootstrap_sample


def perform_bootstrap(dataset):
    n = 10000
    samples = np.empty(shape=(n, len(dataset)))

    for tick in range(n):
        samples[tick] = get_bootstrap_sample(dataset)

    return samples