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


def load_parameters():
    """ Returns constants.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """

    project_directory = get_project_directory()
    filepath = os.path.join(project_directory, 'conf', 'parameters.yml')
    with open(filepath) as f:
        initial_conditions = ruamel.yaml.safe_load(f)

    return initial_conditions
