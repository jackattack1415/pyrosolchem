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


def load_components():
    """ Returns components as a dictionary.
        Outputs
        ---
        components (dict): definitions of components in solution.
    """

    project_directory = get_project_directory()
    filepath = os.sep.join(project_directory + 'conf/components.yml'.split('/'))
    with open(filepath) as f:
        components = yaml.safe_load(f)

    return components


def load_constants():
    """ Returns constants.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """

    project_directory = get_project_directory()
    filepath = os.sep.join(project_directory + 'conf/constants.yml'.split('/'))
    with open(filepath) as f:
        constants = yaml.safe_load(f)

    return constants
