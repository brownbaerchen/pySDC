import numpy as np
from pySDC.core.Problem import ptype


class GenericChebychov(ptype):

    def _get_chebychov_helper(self, *args, **kwargs):
        from pySDC.helpers.problem_helper import ChebychovHelper

        self.cheby = ChebychovHelper(*args, **kwargs)

    def _get_conversion_matrices(self):
        self.T2D = self.cheby.get_conv('T2D')
        self.D2T = self.cheby.get_conv('D2T')
        self.T2U = self.cheby.get_conv('T2U')
        self.U2T = self.cheby.get_conv('U2T')
        self.U2D = self.cheby.get_conv('U2D')
        self.D2U = self.cheby.get_conv('D2U')
