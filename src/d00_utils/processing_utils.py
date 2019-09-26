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


def get_bootstrapped_statistics(data):
    """"""

    samples = perform_bootstrap(data)
    sample_means = np.mean(samples, axis=0)
    sample_avg = np.mean(sample_means)
    sample_std = np.std(sample_means)

    sample_rel_std = sample_std / sample_avg

    return sample_avg, sample_rel_std
