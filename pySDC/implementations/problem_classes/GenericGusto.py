from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh, IMEX_firedrake_mesh
from gusto.core.labels import (
    time_derivative,
    implicit,
    explicit,
    physics_label,
    mass_weighted,
    prognostic,
    transporting_velocity,
)
from firedrake.fml import replace_subject, replace_test_function, Term, all_terms, drop, LabelledForm
import firedrake as fd
import numpy as np


def setup_equation(equation, spatial_methods, transporting_vel='prognostic'):
    """
    Sets up the spatial methods for an equation, by the setting the
    forms used for transport/diffusion in the equation.

    Args:
        equation (:class:`PrognosticEquation`): the equation that the
            transport method is to be applied to.
        spatial_methods: list of spatial methods such as transport or diffusion schemes
    """
    from gusto.core.labels import transport, diffusion
    import logging

    # For now, we only have methods for transport and diffusion
    for term_label in [transport, diffusion]:
        # ---------------------------------------------------------------- #
        # Check that appropriate methods have been provided
        # ---------------------------------------------------------------- #
        # Extract all terms corresponding to this type of term
        residual = equation.residual.label_map(lambda t: t.has_label(term_label), map_if_false=drop)
        variables = [t.get(prognostic) for t in residual.terms]
        methods = list(filter(lambda t: t.term_label == term_label, spatial_methods))
        method_variables = [method.variable for method in methods]
        for variable in variables:
            if variable not in method_variables:
                message = (
                    f'Variable {variable} has a {term_label.label} '
                    + 'term but no method for this has been specified. '
                    + 'Using default form for this term'
                )
                logging.getLogger('problem').warning(message)

    # -------------------------------------------------------------------- #
    # Check that appropriate methods have been provided
    # -------------------------------------------------------------------- #
    # Replace forms in equation
    for method in spatial_methods:
        method.replace_form(equation)

    equation = setup_transporting_velocity(equation, transporting_vel=transporting_vel)
    return equation


def setup_transporting_velocity(equation, transporting_vel='prognostic'):
    """
    Set up the time discretisation by replacing the transporting velocity
    used by the appropriate one for this time loop.
    """
    from firedrake import split
    import ufl

    if transporting_vel == "prognostic":
        # Use the prognostic wind variable as the transporting velocity
        u_idx = equation.field_names.index('u')
        uadv = split(equation.X)[u_idx]
    else:
        uadv = transporting_vel

    equation.residual = equation.residual.label_map(
        lambda t: t.has_label(transporting_velocity),
        map_if_true=lambda t: Term(ufl.replace(t.form, {t.get(transporting_velocity): uadv}), t.labels),
    )

    equation.residual = transporting_velocity.update_value(equation.residual, uadv)

    # Now also replace transporting velocity in the terms that are
    # contained in labels
    for idx, t in enumerate(equation.residual.terms):
        if t.has_label(transporting_velocity):
            for label in t.labels.keys():
                if type(t.labels[label]) is LabelledForm:
                    t.labels[label] = t.labels[label].label_map(
                        lambda s: s.has_label(transporting_velocity),
                        map_if_true=lambda s: Term(ufl.replace(s.form, {s.get(transporting_velocity): uadv}), s.labels),
                    )

                    equation.residual.terms[idx].labels[label] = transporting_velocity.update_value(
                        t.labels[label], uadv
                    )
    return equation


class GenericGusto(Problem):
    dtype_u = firedrake_mesh
    dtype_f = firedrake_mesh
    rhs_n_labels = 1

    def __init__(
        self,
        equation,
        apply_bcs=True,
        solver_parameters=None,
        stop_at_divergence=False,
        LHS_cache_size=12,
        residual=None,
        *active_labels,
    ):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        # TODO: documentation of __init__

        self.equation = equation
        self.residual = equation.residual if residual is None else residual
        self.field_name = equation.field_name
        self.fs = equation.function_space
        self.idx = None
        if solver_parameters is None:
            # default solver parameters
            solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        self.solver_parameters = solver_parameters
        self.stop_at_divergence = stop_at_divergence

        # -------------------------------------------------------------------- #
        # Setup caches
        # -------------------------------------------------------------------- #

        self.x_out = fd.Function(self.fs)
        self.solvers = {}
        self._u = fd.Function(self.fs)

        super().__init__(self.fs)
        self._makeAttributeAndRegister('LHS_cache_size', 'apply_bcs', localVars=locals(), readOnly=True)
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['ksp'] = WorkCounter()
        self.work_counters['solver_setup'] = WorkCounter()
        self.work_counters['solver'] = WorkCounter()

    @property
    def bcs(self):
        if not self.apply_bcs:
            return None
        else:
            return self.equation.bcs[self.equation.field_name]

    def invert_mass_matrix(self, rhs):
        self._u.assign(rhs.functionspace)

        if 'mass_matrix' not in self.solvers.keys():
            mass_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )
            rhs_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self._u, old_idx=self.idx),
                map_if_false=drop,
            )

            problem = fd.NonlinearVariationalProblem((mass_form - rhs_form).form, self.x_out, bcs=self.bcs)
            solver_name = self.field_name + self.__class__.__name__
            self.solvers['mass_matrix'] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup']()

        self.solvers['mass_matrix'].solve()

        return self.dtype_u(self.x_out)

    def eval_f(self, u, *args):
        self._u.assign(u.functionspace)

        if 'eval_rhs' not in self.solvers.keys():
            residual = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=replace_subject(self._u, old_idx=self.idx),
                map_if_true=drop,
            )
            mass_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )

            problem = fd.NonlinearVariationalProblem((mass_form + residual).form, self.x_out, bcs=self.bcs)
            solver_name = self.field_name + self.__class__.__name__
            self.solvers['eval_rhs'] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup']()

        self.solvers['eval_rhs'].solve()
        self.work_counters['rhs']()

        return self.dtype_f(self.x_out)

    def solve_system(self, rhs, factor, u0, *args):
        self.x_out.assign(u0.functionspace)  # set initial guess
        self._u.assign(rhs.functionspace)

        if factor not in self.solvers.keys():
            if len(self.solvers) >= self.LHS_cache_size + self.rhs_n_labels:
                self.solvers.pop(
                    [me for me in self.solvers.keys() if type(me) in [float, int, np.float64, np.float32]][0]
                )

            # setup left hand side (M - factor*f)(u)
            # put in output variable
            residual = self.residual.label_map(all_terms, map_if_true=replace_subject(self.x_out, old_idx=self.idx))
            # multiply f by factor
            residual = residual.label_map(
                lambda t: t.has_label(time_derivative), map_if_false=lambda t: fd.Constant(factor) * t
            )

            # subtract right hand side
            mass_form = self.residual.label_map(lambda t: t.has_label(time_derivative), map_if_false=drop)
            residual -= mass_form.label_map(all_terms, map_if_true=replace_subject(self._u, old_idx=self.idx))

            # construct solver
            problem = fd.NonlinearVariationalProblem(residual.form, self.x_out, bcs=self.bcs)
            solver_name = f'{self.field_name}-{self.__class__.__name__}-{factor}'
            self.solvers[factor] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup']()

        try:
            self.solvers[factor].solve()
        except fd.exceptions.ConvergenceError as error:
            if self.stop_at_divergence:
                raise error
            else:
                self.logger.debug(error)

        self.work_counters['ksp'].niter += self.solvers[factor].snes.getLinearSolveIterations()
        self.work_counters['solver']()
        return self.dtype_u(self.x_out)


class GenericGustoImex(GenericGusto):
    dtype_f = IMEX_firedrake_mesh
    rhs_n_labels = 2

    def evaluate_individual_term(self, u, label):
        self._u.assign(u.functionspace)

        if label not in self.solvers.keys():
            residual = self.residual.label_map(
                lambda t: t.has_label(label) and not t.has_label(time_derivative),
                map_if_true=replace_subject(self._u, old_idx=self.idx),
                map_if_false=drop,
            )
            mass_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )

            problem = fd.NonlinearVariationalProblem((mass_form + residual).form, self.x_out, bcs=self.bcs)
            solver_name = self.field_name + self.__class__.__name__
            self.solvers[label] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup'] = WorkCounter()

        self.solvers[label].solve()
        return self.x_out

    def eval_f(self, u, *args):
        me = self.dtype_f(self.init)
        me.impl.assign(self.evaluate_individual_term(u, implicit))
        me.expl.assign(self.evaluate_individual_term(u, explicit))
        self.work_counters['rhs']()
        return me

    def solve_system(self, rhs, factor, u0, *args):
        self.x_out.assign(u0.functionspace)  # set initial guess
        self._u.assign(rhs.functionspace)

        if factor not in self.solvers.keys():
            if len(self.solvers) >= self.LHS_cache_size + self.rhs_n_labels:
                self.solvers.pop(
                    [me for me in self.solvers.keys() if type(me) in [float, int, np.float64, np.float32]][0]
                )

            # setup left hand side (M - factor*f_I)(u)
            # put in output variable
            residual = self.residual.label_map(
                lambda t: t.has_label(time_derivative) or t.has_label(implicit),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )
            # multiply f_I by factor
            residual = residual.label_map(
                lambda t: t.has_label(implicit) and not t.has_label(time_derivative),
                map_if_true=lambda t: fd.Constant(factor) * t,
            )

            # subtract right hand side
            mass_form = self.residual.label_map(lambda t: t.has_label(time_derivative), map_if_false=drop)
            residual -= mass_form.label_map(all_terms, map_if_true=replace_subject(self._u, old_idx=self.idx))

            # construct solver
            problem = fd.NonlinearVariationalProblem(residual.form, self.x_out, bcs=self.bcs)
            solver_name = f'{self.field_name}-{self.__class__.__name__}-{factor}'
            self.solvers[factor] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup'] = WorkCounter()

        self.solvers[factor].solve()
        try:
            self.solvers[factor].solve()
        except fd.exceptions.ConvergenceError as error:
            if self.stop_at_divergence:
                raise error
            else:
                self.logger.debug(error)

        self.work_counters['ksp'].niter += self.solvers[factor].snes.getLinearSolveIterations()
        self.work_counters['solver']()
        return self.dtype_u(self.x_out)
