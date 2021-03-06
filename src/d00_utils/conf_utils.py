import os
import ruamel.yaml


def get_project_directory():
    """ Returns project directory.

        :return: str. path back to top level directory
    """

    project_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

    return project_directory


def load_compounds():
    """ Returns compounds as two dictionaries, split into water and other chemical compounds.

        :return: dict. definitions of components in solution.
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


def load_experiments(file_name, experiment_names=None):
    """ Returns experimental parameters for experiment names of choice.

        :param file_name: str. Only the file name (not the path) of the yml file, located in conf folder.
        :param experiment_names: list. List of the experiment names to be loaded. None grabs all experiments.
        :return: dict. Contains the dict embedded in the yml of the file name selected.
    """

    project_directory = get_project_directory()
    file_path = os.path.join(project_directory, 'conf', file_name)
    with open(file_path) as f:
        experiments = ruamel.yaml.safe_load(f)

    if experiment_names is None:
        experiments = experiments
    elif experiment_names is not None:
        experiments = dict((exp, experiments[exp]) for exp in experiment_names)

    return experiments
