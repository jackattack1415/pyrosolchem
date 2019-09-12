import os
import yaml

def get_project_directory():
    """ Returns project directory.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """

    project_directory = os.path.dirname(os.path.realpath(__file__))

    return project_directory


def load_compounds():
    """ Returns components as a dictionary.
        Outputs
        ---
        components (dict): definitions of components in solution.
    """

    project_directory = get_project_directory()
    filepath = os.sep.join(project_directory + 'conf/compounds.yml'.split('/'))
    with open(filepath) as f:
        compounds = yaml.safe_load(f)

    return compounds


def load_parameters():
    """ Returns constants.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """

    project_directory = get_project_directory()
    filepath = os.sep.join(project_directory + 'conf/parameters.yml'.split('/'))
    with open(filepath) as f:
        initial_conditions = yaml.safe_load(f)

    return initial_conditions
