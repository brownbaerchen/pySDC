#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:39:30 2023

@author: telu
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve, cg

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


class GenericNDimFFTLaplacian(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=512,
        coeff=1.0,
        freq=1.0,
        coeff=1.0,
        lintol=1e-12,
        liniter=10000,
        solver_type='direct',
    ):
        # make sure parameters have the correct types
        if not type(nvars) in [int, tuple]:
            raise ProblemError('nvars should be either tuple or int')

        # transforms nvars into a tuple
        if type(nvars) is int:
            nvars = (nvars,)

        # automatically determine ndim from nvars
        ndim = len(nvars)
        if ndim > 2:
            raise ProblemError(f'can work with up to three dimensions, got {ndim}')

        # eventually extend freq to other dimension
        if type(freq) is int:
            freq = (freq,) * ndim
        if len(freq) != ndim:
            raise ProblemError(f'len(freq)={len(freq)}, different to ndim={ndim}')

        # invoke super init, passing number of dofs
        super().__init__(init=(nvars[0] if ndim == 1 else nvars, None, np.dtype('float64')))

        # compute dx (equal in both dimensions) and get discretization matrix A
        dx = 1.0 / nvars[0]
        xvalues = np.array([i * dx for i in range(nvars[0])])

        self.xvalues = xvalues
        self.Id = sp.eye(np.prod(nvars), format='csc')

        # store attribute and register them as parameters
        self._makeAttributeAndRegister('nvars', 'stencil_type', 'order', 'bc', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('freq', 'lintol', 'liniter', 'solver_type', localVars=locals())

        if self.solver_type != 'direct':
            self.work_counters[self.solver_type] = WorkCounter()

        self.generate_laplacian(coeff)

    @property
    def ndim(self):
        """Number of dimensions of the spatial problem"""
        return len(self.nvars)

    @property
    def dx(self):
        """Size of the mesh (in all dimensions)"""
        return self.xvalues[1] - self.xvalues[0]

    @property
    def grids(self):
        """ND grids associated to the problem"""
        x = self.xvalues
        if self.ndim == 1:
            return x
        if self.ndim == 2:
            return x[None, :], x[:, None]
        if self.ndim == 3:
            return x[None, :, None], x[:, None, None], x[None, None, :]

    def generate_laplacian(self):
        raise NotImplementedError('Please implement a function for the laplacian!')

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            The RHS values.
        """
        f = self.f_init
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        sol : dtype_u
            The solution of the linear solver.
        """
        raise NotImplementedError('Please implement a function for solving linear systems!')


class Generic1DimFFTLaplacian(GenericNDimFFTLaplacian):
    def __init__(
        self,
        **kwagrs,
    ):
        super().__init__(self, **kwagrs)

        if self.ndim != 1:
            raise ProblemError(f'This is one dimensional FFT Laplacian, you are trying to do {self.ndim} dimensions')

    def generate_laplacian(self, coeff):
        kx = np.zeros(self.init[0] // 2 + 1)
        for i in range(0, len(kx)):
            kx[i] = 2 * np.pi / self.params.L * i

        self.ddx = kx * 1j
        self.lap = -(kx**2) * coeff

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        tmp = np.fft.rfft(rhs) / (1.0 - factor * self.lap)
        me[:] = np.fft.irfft(tmp)

        return me
