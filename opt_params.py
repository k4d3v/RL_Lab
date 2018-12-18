""" For storing optimization parameters"""


class Opt:
    def __init__(self, length=150, MFEPLS = 30, verbosity = 1):
        """

        :param length: Max. number of line searches
        :param MFEPLS: Max. number of fun evaluations per line search
        :param verbosity: Specifies how much info is displayed during policy learning; Options: 0-3
        """
        self.length = length
        self.MFEPLS = MFEPLS
        self.verbosity = verbosity
        self.fh = 0