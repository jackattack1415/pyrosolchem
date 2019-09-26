import os
import ruamel.yaml


def get_project_directory():
    """ Returns project directory.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """

    project_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

    return project_directory


def load_compounds():
    """ Returns components as a dictionary.
        Outputs
        ---
        components (dict): definitions of components in solution.
    """

    project_directory = get_project_directory()
    filepath = os.path.join(project_directory, 'conf', 'compounds.yml')
    with open(filepath) as f:
        compounds = ruamel.yaml.safe_load(f)

    water = compounds['water']
    del compounds['water']

    return compounds, water


def load_constants():
    """ Returns constants.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """

    project_directory = get_project_directory()
    filepath = os.path.join(project_directory, 'conf', 'constants.yml')
    with open(filepath) as f:
        constants = ruamel.yaml.safe_load(f)

    return constants


def load_experiments(experiment_names=None):
    """ Returns experimental parameters for experiment names of choice.
    """

    project_directory = get_project_directory()
    filepath = os.path.join(project_directory, 'conf', 'experiments.yml')
    with open(filepath) as f:
        experiments = ruamel.yaml.safe_load(f)

    if experiment_names is None:
        experiments = experiments
    elif experiment_names is not None:
        experiments = dict((exp, experiments[exp]) for exp in experiment_names)

    return experiments


def load_paths():
    """ Returns constants.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """

    project_directory = get_project_directory()
    filepath = os.path.join(project_directory, 'conf', 'paths.yml')
    with open(filepath) as f:
        paths = ruamel.yaml.safe_load(f)

    return paths