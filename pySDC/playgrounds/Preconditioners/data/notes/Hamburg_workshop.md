# Notes for parallel SDC Workshop at TUHH in December 2022

## Reminder: Parallelize SDC accross the method
We want to execute a single SDC iteration in parallel rather than multiple iterations or steps or whatever simultaneously (See Ruth's projects for that...)
SDC solves the collocation problem, where we integrate both sides of an initial value problem and discretize the integral using a quadrature rule:

$$
u(t) = u_0 + \int_0^{t}f(u(\tau))d\tau \rightarrow u_m = u_0 + \Delta t\sum_{j=1}^M q_{m,j}f(u_j) \rightarrow (I - \Delta t QF)(\vec{u})=\vec{u}_0
$$

Then, we construct a defect equation which we solve with "something simpler"

$$
\delta - \int_0^t(f(u+\delta) - f(u))d\tau = r,
$$

with the residual

$$
r = u_0 + \int_0^t f(u) d\tau - u.
$$

When discretizing this, we choose a high order quadrature rule with few stages $Q$, which is dense, to compute the residual and a low order, lower triangular quadrature rule $Q_\Delta$ for solving the defect equation.
Putting everything together, we get for the update at each node:

$$
u^{k+1}_{m+1} - \Delta t \tilde{q}_{m+1, m+1}f(u^{k+1}_{m+1}) = u_0 + \Delta t \sum_{j=1}^{m}\tilde{q}_{m+1, j}f(u_j^{k+1}) + \Delta t \sum_{j+1}^M \left(q_{m+1, j}-\tilde{q}_{m+1, j}\right)f(u^k_j),
$$

where the lower triangular nature of the preconditioner becomes apparent in the first sum on the right hand side, which only goes to $m$ rather than $M$.
Now this sum realizes the forward substitution for Gauss-Seidel like preconditioners, but we can parallelize the iteration by choosing a Jacobi-like preconditioner, which is diagonal and hence this sum disappears.

## Reminder: Adaptivity
Simple recipy:
 - Construct an embedded method by computing two solutions of different order $u^{(p)}$ and $u^{(p+1)}$
 - Estimate the error of the lower order method by subtracting these:

 $$
\epsilon = \|u^{(p)} - u^{(p_1)}\| = \|u^{(p)} - u^* - (u^{(p+1)} - u^*)\| = \|e^{(p)} - e^{(p+1)}\| = e^{(p)} + \mathcal{O}(\Delta t^{p+2})
 $$
  - Compute an "optimal" step size to reach a tolerance $\epsilon_\mathrm{TOL}$:

  $$
\Delta t^* = 0.9 \Delta t \left(\frac{\epsilon_\mathrm{TOL}}{\epsilon}\right)^{1/p}
  $$

  - Move on to next step if $\epsilon \leq \epsilon_\mathrm{TOL}$ or restart step with smaller step size otherwise

## How to get diagonal preconditioners with adaptivity?
We can run a problem with adaptivity, which ensures the resolution we want and check how many iterations we needed.
Then, we can minimize the number of iterations by allowing some scipy algorithm to change the values on the diagonal of the preconditioner.

We choose a range of Dahlquist problems with parameters representing eigenvalues of finite difference stencils for advection and diffusion, which we actually want to solve.

<p>
<img src="./Hamburg_workshop_plots/eigenvalues.png" alt="Problem parameters for the Dahlquist problems" style="width:40%;"/>
<em>Problem parameters for the range of Dahlquist problems we use for optimization representing eigenvalues of finite difference discretizations of advection and diffusion problems.
</em>
</p>

We evaluate the quality of preconditioners by checking how well it can cope with stiff problems.
Since the best thing we have at the moment is the MIN preconditioner from Robert's paper, we use that as initial conditions for the optimization.

<p>
<img src="./Hamburg_workshop_plots/stiffness-spread.png" alt="Stiffness plot" style="width:100%;"/>
<em>We just end up at the MIN preconditioner when plugging that into the optimization, so it can handle stiff ploblems equally well.
Notice MIN3, but don't worry about it.
Some obscure solver generated this, but we don't know how and hence Robert can't publish it.
</em>
</p>

So how do the preconditioners actually look?

<p>
<img src="./Hamburg_workshop_plots/weights.png" alt="Weights of the preconditioners" style="width:40%;"/>
<em>Visualization of integration weights of the preconditioners.
Notice the whichcraft in everything but implicit Euler.
</em>
</p>

What does that mean for convergence?
Well, except for implicit Euler, the preconditioners have been selected because they can provide fast convergence, but it turns out only if the initial guess is sufficiently close to the correct solution.

<p>
<img src="./Hamburg_workshop_plots/order-LU-advection.png" alt="Order in time with the LU preconditioner." style="width:70%;"/>
<em>Order in time with the LU preconditioner.
The left and right panels differ only in the initial guess used to initialize the collocation nodes.
Notice that we lose an order if we don't spread the initial conditions to the nodes.
</em>
</p>

This order reduction has possible consequences on adaptivity: If we assume too high of an order, we are going to compute a step size that is too large and cause restarts.

Looking at the weights, we notice that many preconditions integrate over too short of an interval.
Could the result be something like successive over-relaxation, where the correction is artifically enlarged?
I need to think a little bit more if this is actually the case, but we can just optimize a single rescaling parameter for the preconditioners.
Starting again at MIN, we find an optimal rescaling of 1.25 and the stiffness plot does not look half bad actually.

<p>
<img src="./Hamburg_workshop_plots/stiffness-SOR.png" alt="Stiffness plot" style="width:100%;"/>
<em>Diagonal is MIN rescaled by a factor 1.25.
We can almost rival the MIN3 preconditioner with this configuration.
</em>
</p>

Does this make sense?
Not necessarily...
I can offer no explanation why this scheme might work better than MIN...
