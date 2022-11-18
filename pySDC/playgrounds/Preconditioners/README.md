# Preconditioners
The major goal of this project is to derive preconditioners using optimization and adaptivity.

We made some considerations of how to evaluate the contraction factors of preconditioners using Fourier transform [here](data/notes/Fourier.md).

We discussed details of the optimization [here](data/notes/optimization.md).

## Things we learned so far
 - Preconditioners are better at handling stiff problems when we initialized the intermediate solutions during optimization randomly.

 ## Questions
 - Is the MIN preconditioner successive under-relaxation for a Jacobi-type iteration? I.e. does rescaling the integration boundary result directly in rescaling the increment?
 - What is the impact of the residual in the defect equation? This alone leads to convergence, albeit much slower, but I think I need to take this into account.

## TODOs
 - Make more elaborate objective functions
 - Can we use the extrapolation error estimate in objective functions?
 - Use Dahlquist problem for optimization