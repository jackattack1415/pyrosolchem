import os


def get_project_directory():
    """ Returns project directory.
        Outputs
        ---
        project_directory (str): path back to top level directory
    """
    project_directory = os.path.dirname(os.path.realpath(__file__))

    return project_directory