from timeit import default_timer as timer
from library import *
from sparse_reg import *


def identify_equations(Q, reg_opts, library, observables, threshold=1e-5, min_complexity=1,
                       max_complexity=None, max_equations=999, timed=True, excluded_terms=None, multipliers=None):
    """
    does not properly identify all implications since generation via dx is incomplete from an indexing perspective
    e.g., both generation via differentiation and multiplication cannot follow the path v_i->dj v_i->dj^2 v_i
    most probable fix is by considering indexed tensors of ranks >=2, but this bug isn't critical
    """
    if timed:
        start = timer()
    equations = []
    lambdas = []
    derived_eqns = {}
    if excluded_terms is None:
        excluded_terms = set()
    # this can be eliminated by keeping track of two different max_complexities in args
    lib_max_complexity = max([term.complexity for term in library])  # generate list of derived terms up to here
    if max_complexity is None:
        max_complexity = lib_max_complexity
    if multipliers is None:
        obs_terms = set([obs_to_term(obs) for obs in observables])
        multipliers = set(get_multipliers(obs_terms, lib_max_complexity))
    for complexity in range(min_complexity, max_complexity + 1):
        while len(equations) < max_equations:
            selection = [(term, i) for (i, term) in enumerate(library) if term.complexity <= complexity
                         and term not in excluded_terms]
            sublibrary = [s[0] for s in selection]
            inds = [s[1] for s in selection]
            reg_opts['subinds'] = inds
            # identify model
            eq, res = make_equation_from_Xi(*sparse_reg(Q, **reg_opts), sublibrary)
            if res > threshold:
                break
            equations.append(eq)
            lambdas.append(res)
            # add some output about the discovered model
            if timed:
                # noinspection PyUnboundLocalVariable
                time = timer() - start
                print(f"[{time:.2f} s]")
            print(f'Identified model: {eq} (order {complexity}, residual {res:.2e})')
            # eliminate terms via infer_equations
            derived_eqns[str(eq)] = []
            for new_eq in infer_equations(eq, multipliers, lib_max_complexity):
                lhs, rhs = new_eq.eliminate_complex_term()
                # print(lhs)
                excluded_terms.add(lhs)
                derived_eqns[str(eq)].append(form_equation(lhs, rhs))
    return equations, lambdas, derived_eqns, excluded_terms


def interleave_identify(Qs, reg_opts_list, libraries, observables, threshold=1e-5, min_complexity=1,
                        max_complexity=None, max_equations=999, timed=True, excluded_terms=None):
    equations = []
    lambdas = []
    derived_eqns = {}
    if excluded_terms is None:
        excluded_terms = set()
    lib_max_complexity = max([term.complexity for library in libraries for term in library])
    if max_complexity is None:
        max_complexity = lib_max_complexity
    obs_terms = set([obs_to_term(obs) for obs in observables])
    multipliers = set(get_multipliers(obs_terms, lib_max_complexity))
    for complexity in range(min_complexity, max_complexity + 1):
        for Q, reg_opts, library in zip(Qs, reg_opts_list, libraries):
            eqs_i, lbds_i, der_eqns_i, exc_terms_i = identify_equations(Q, reg_opts, library,
                                                                        observables, threshold=threshold,
                                                                        min_complexity=complexity,
                                                                        max_complexity=complexity,
                                                                        max_equations=max_equations, timed=timed,
                                                                        excluded_terms=excluded_terms,
                                                                        multipliers=multipliers)
            equations += eqs_i
            lambdas += lbds_i
            derived_eqns.update(der_eqns_i)
            excluded_terms.update(exc_terms_i)
    return equations, lambdas, derived_eqns, excluded_terms


def make_equation_from_Xi(Xi, lambd, best_term, lambda1, sublibrary):
    if lambda1 < lambd:
        return Equation([sublibrary[best_term]], [1]), lambda1
    else:
        zipped = [(sublibrary[i], c) for i, c in enumerate(Xi) if c != 0]
        return Equation([e[0] for e in zipped], [e[1] for e in zipped]), lambd


def infer_equations(equation, multipliers, max_complexity):
    yield equation
    complexity = max([term.complexity for term in equation.term_list])
    if complexity >= max_complexity:  # should be at most equal actually
        return
    rem_complexity = max_complexity - complexity
    eq_dt = equation.dt()
    eq_dx = equation.dx()
    yield from infer_equations(eq_dt, multipliers, max_complexity)
    yield from infer_equations(eq_dx, multipliers, max_complexity)
    for term in multipliers:
        if term.complexity <= rem_complexity:
            yield from infer_equations(term * equation, multipliers, max_complexity)


def form_equation(lhs, rhs):
    if rhs is None:
        return Equation([lhs], [1])
    else:
        return -1 * TermSum([lhs]) + rhs


def obs_to_term(observable):
    prim = LibraryPrimitive(DerivativeOrder(0, 0), observable)
    tensor = LibraryTensor(prim)
    labels = {k: [1] for k in list(range(tensor.rank))}
    return LibraryTerm(tensor, labels=labels)


def get_multipliers(observables, max_complexity):
    if max_complexity <= 0:
        return
    frontier = []
    for obs in observables:
        yield obs
        if max_complexity > 1:
            frontier.append(obs.dt().to_term())
            frontier.append(obs.dx().to_term())
    yield from get_multipliers(frontier, max_complexity - 1)


def rebalance(lhs, rhs):
    if len(lhs.term_list) > 1:  # note that we don't need to check =0 since 1 can't be the most complex term
        lhs1, lhs2, nrm = lhs.eliminate_complex_term(return_normalization=True)
        if rhs is None:
            new_rhs = -1 * lhs2
        else:
            new_rhs = (1 / nrm) * rhs + (-1 * lhs2)
        return lhs1, new_rhs
    else:
        return lhs.to_term(), rhs
