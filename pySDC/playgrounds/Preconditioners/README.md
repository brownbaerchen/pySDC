# Preconditioners
The major goal of this project is to derive preconditioners using optimization and adaptivity.

We made some considerations of how to evaluate the contraction factors of preconditioners using Fourier transform [here](data/notes/Fourier.md).

We discussed details of the optimization [here](data/notes/optimization.md).

In an attempt to understand what the MIN preconditioner is doing, I looked a little bit into successive over-relaxation for SDC [here](data/notes/SOR.md).
In my personal opinion, this topic is worthy of further exploration, but it is not really connected to the optimization and adaptivity part and so I don't know if I will find the time.

## Things we learned so far
 - Preconditioners are better at handling stiff problems when we initialized the intermediate solutions during optimization randomly.
 - Chosing a suitable region of Dahlquist problems is better than choosing a problem directly, although what this suitable region is is tbd.

 ## Questions
 - Is the MIN preconditioner successive under-relaxation for a Jacobi-type iteration? I.e. does rescaling the integration boundary result directly in rescaling the increment?
 - Is the MIN preconditioner really successive *over*-relaxation, because the defect is integrated over a shorter time frame but with the same right hand side, so we compute a larger correction?
 - What is the impact of the residual in the defect equation? This alone leads to convergence, albeit much slower, but I think I need to take this into account.

## TODOs
 - Make more elaborate objective functions
 - Can we use the extrapolation error estimate in objective functions?
 - Use complex elements in the preconditioners. However, this will result in some positive real Dahlquist parameters, so probably this will just yield BS
 - Try optimizing lower triangluar preconditioners rather than just diagonal ones