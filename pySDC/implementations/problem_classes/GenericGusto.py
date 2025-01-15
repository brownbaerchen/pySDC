from pySDC.core.problem import Problem
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
    rhs_labels = []
    lhs_labels = []

    def __init__(self, equation, apply_bcs=True, solver_parameters=None, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        self.equation = equation
        self.residual = equation.residual
        self.field_name = equation.field_name
        self.fs = equation.function_space
        self.idx = None
        if solver_parameters is None:
            # default solver parameters
            solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        self.solver_parameters = solver_parameters

        if len(active_labels) > 0:
            self.residual = self.residual.label_map(
                lambda t: any(t.has_label(time_derivative, *active_labels)), map_if_false=drop
            )

        self.evaluate_source = []
        self.physics_names = []
        for t in self.residual:
            if t.has_label(physics_label):
                physics_name = t.get(physics_label)
                if t.labels[physics_name] not in self.physics_names:
                    self.evaluate_source.append(t.labels[physics_name])
                    self.physics_names.append(t.labels[physics_name])

        # Check if there are any mass-weighted terms:
        if len(self.residual.label_map(lambda t: t.has_label(mass_weighted), map_if_false=drop)) > 0:
            for field in equation.field_names:

                # Check if the mass term for this prognostic is mass-weighted
                if (
                    len(
                        self.residual.label_map(
                            (
                                lambda t: t.get(prognostic) == field
                                and t.has_label(time_derivative)
                                and t.has_label(mass_weighted)
                            ),
                            map_if_false=drop,
                        )
                    )
                    == 1
                ):

                    field_terms = self.residual.label_map(
                        lambda t: t.get(prognostic) == field and not t.has_label(time_derivative), map_if_false=drop
                    )

                    # Check that the equation for this prognostic does not involve
                    # both mass-weighted and non-mass-weighted terms; if so, a split
                    # timestepper should be used instead.
                    if len(field_terms.label_map(lambda t: t.has_label(mass_weighted), map_if_false=drop)) > 0:
                        if len(field_terms.label_map(lambda t: not t.has_label(mass_weighted), map_if_false=drop)) > 0:
                            raise ValueError(
                                'Mass-weighted and non-mass-weighted terms are present in a '
                                + f'timestepping equation for {field}. As these terms cannot '
                                + 'be solved for simultaneously, a split timestepping method '
                                + 'should be used instead.'
                            )
                        else:
                            # Replace the terms with a mass_weighted label with the
                            # mass_weighted form. It is important that the labels from
                            # this new form are used.
                            self.residual = self.residual.label_map(
                                lambda t: t.get(prognostic) == field and t.has_label(mass_weighted),
                                map_if_true=lambda t: t.get(mass_weighted),
                            )
        self.idx = None

        # -------------------------------------------------------------------- #
        # Make boundary conditions
        # -------------------------------------------------------------------- #

        if not apply_bcs:
            self.bcs = None
        else:
            self.bcs = equation.bcs[equation.field_name]

        # -------------------------------------------------------------------- #
        # Setup caches
        # -------------------------------------------------------------------- #

        self.x_out = fd.Function(self.fs)
        self.solvers = {}
        self._u = fd.Function(self.fs)

        super().__init__(self.fs)

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

        self.solvers['eval_rhs'].solve()

        return self.dtype_f(self.x_out)

    def solve_system(self, rhs, factor, u0, *args):
        self.x_out.assign(u0.functionspace)  # set initial guess
        self._u.assign(rhs.functionspace)

        if factor not in self.solvers.keys():
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

            problem = fd.NonlinearVariationalProblem(residual.form, self.x_out, bcs=self.bcs)
            solver_name = f'{self.field_name}-{self.__class__.__name__}-{factor}'
            self.solvers[factor] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )

        self.solvers[factor].solve()
        return self.dtype_u(self.x_out)


class GenericGustoImex(GenericGusto):
    dtype_f = IMEX_firedrake_mesh

    def evaluate_terms_IMEX(self, u, label):
        self._u.assign(u.functionspace)

        if label not in self.solvers.keys():
            residual = self.residual.label_map(
                lambda t: t.has_label(label), map_if_true=replace_subject(self._u, old_idx=self.idx), map_if_false=drop
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

        self.solvers[label].solve()
        return self.x_out

    def eval_f(self, u, *args):
        me = self.dtype_f(self.init)
        if self.imex:
            me.impl.assign(self.evaluate_terms_IMEX(u, implicit))
            me.expl.assign(self.evaluate_terms_IMEX(u, explicit))
        else:
            me.impl.assign(self.evaluate_terms_fully_implicit(u))
            me.expl.assign(0)
        return me

    def solve_system(self, rhs, factor, u0, *args):
        self.x_out.assign(u0.functionspace)  # set initial guess
        self._u.assign(rhs.functionspace)

        mass_form = self.residual.label_map(lambda t: t.has_label(time_derivative), map_if_false=drop)

        if factor not in self.solvers.keys():
            residual = mass_form.label_map(all_terms, map_if_true=replace_subject(self.x_out, old_idx=self.idx))
            raise NotImplementedError

            problem = fd.NonlinearVariationalProblem(residual.form, self.x_out, bcs=self.bcs)
            solver_name = f'{self.field_name}-{self.__class__.__name__}-{factor}'
            self.solvers[factor] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )

        self.solvers[factor].solve()
        return self.dtype_u(self.x_out)
