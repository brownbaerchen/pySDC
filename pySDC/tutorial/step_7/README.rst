Step-7: pySDC with external libraries
=====================================

pySDC can be used with external libraries, in particular for spatial discretization, parallelization and solving of linear and/or nonlinear systems.
In the following, we show a few examples of pySDC + X.

Part A: pySDC and FEniCS
------------------------

In this example, pySDC is coupled with the `FEniCS framework <https://fenicsproject.org/>`_ for using finite elements in space.
This implies significant changes to the algorithm, depending on whether or not the mass matrix should be inverted.
SDC, MLSDC and PFASST can be used without changes when the right-hand side of the ODE is defined with the inverse of the mass matrix.
Otherwise, the mass matrix has to be used for e.g. the tau-correction.
This example tests different variants of this methodology for SDC, MLSDC and PFASST.

Important things to note:

- This example shows that even core routines like the `BaseTransfer` can be overwritten if needed.
- It is also valuable to check out the data type and transfer classes required to work with FEniCS. Both can be found in the `implementations` folder.

.. include:: doc_step_7_A.rst

Part B: mpi4py-fft for parallel Fourier transforms
--------------------------------------------------

The most prominent parallel solver is, probably, the FFT.
While many implementations or wrappers for Python exist, we decided to use `mpi4py-fft <https://mpi4py-fft.readthedocs.io/en/latest/>`_, which provided the easiest installation, a simple API and good parallel scaling.
As an example, we here test the nonlinear Schrödinger equation, using the IMEX sweeper to treat the nonlinear parts explicitly.
The code allows to work both in real and spectral space, while the latter is usually faster.
This example tests SDC, MLSDC and PFASST.

Important things to note:

- The code runs both in serial using just `python B_pySDC_with_mpi4pyfft.py` and also in parallel using `mpirun -np 2 python B_pySDC_with_mpi4pyfft.py`.
- The nonlinear Schrödinger example is not expected to work well with PFASST. In fact, SDC and MLSDC converge for larger time-steps, but PFASST does not.

.. include:: doc_step_7_B.rst


Part C: Time-parallel pySDC with space-parallel PETSc
-----------------------------------------------------

With rather unfavorable scaling properties, parallel-in-time methods are only really useful when spatial parallelization is maxed out.
To work with spatial parallelization, this part shows how to (1) include and work with an external library and (2) set up space- and time-parallel runs.
We use again the forced heat equation as our testbed and `PETSc <http://www.mcs.anl.gov/petsc/>`_ for the space-parallel data structures and linear solver.
See `implementations/datatype_classes/petsc_dmda_grid.py` and `implementations/problem_classes/HeatEquation_2D_PETSc_forced.py` for details on the PETSc bindings.

Important things to note:

- We need processors in space and time, which can be achieved by `comm.Split` and coloring. The space-communicator is then passed to the problem class.
- Below, we run the code 3 times: with 1 and 2 processors in space as well as 4 processors (2 in time and 2 in space). Do not expect scaling due to the CI environment.

.. include:: doc_step_7_C.rst


Part D: pySDC and PyTorch
-------------------------

PyTorch is a library for machine learning. The data structure is called tensor and allows to run on CPUs as well as GPUs in addition to access to various machine learning methods.
Since the potential for use in pySDC is very large, we have started on a datatype that allows to use PyTorch tensors throughout pySDC.

This example trains a network to predict the results of implicit Euler solves for the heat equation. It is too simple to do anything useful, but demonstrates how to use tensors in pySDC and then apply the enormous PyTorch infrastructure.
This is work in progress in very early stages! The tensor datatype is the simplest possible implementation, rather than an efficient one.
If you want to work on this, your input is appreciated!

.. include:: doc_step_7_D.rst


Part E: pySDC and Firedrake
---------------------------

`Firedrake <https://github.com/firedrakeproject/firedrake>`_ is a finite element library with similar features as FEniCS.
The below example runs the same heat equation as in the FEniCS example, but implemented in Firedrake.
See the problem class implementation as a blueprint for how to implement problems with Firedrake in a way that pySDC can understand: `pySDC/implementations/problem_classes/HeatFiredrake.py <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/problem_classes/HeatFiredrake.py>`_

.. include:: doc_step_7_E.rst


Part F: pySDC and Gusto
---------------------------

`Gusto <https://github.com/firedrakeproject/gusto>`_ is a toolkit for geophysical simulations that uses `Firedrake <https://github.com/firedrakeproject/firedrake>`_ for spatial discretization.
The below example is an adaptation of the Williamson 5 test case as implemented in Gusto.
This coupling works slightly different to the other examples within this tutorial, as timestepping is part of Gusto.
The aim of the coupling is not a spatial discretization, but to use the equations that are implemented in Gusto.
A Gusto equation includes the basic form of the equation set, but a crucial part is to modify terms in the discretized equations with spatial methods, such as upwinding schemes.
We get the finished equation set into pySDC by setting up pySDC as a time discretization for Gusto and instantiating a Gusto timestepper.
During this instantiation the equation, and the residual that is used for solving systems, is modified with all the spatial methods.
Afterwards, you have a Gusto timestepping scheme that you can run in Gusto and a pySDC controller that you can run by itself.

.. include:: doc_step_7_F.rst
