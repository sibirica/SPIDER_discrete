# It may or may not be nicer to take the SRDataset object as input for some of these
from timeit import default_timer as timer

from library import *
from sparse_reg import *


# does not properly identify all implications since generation via dx is incomplete from an indexing perspective
# e.g., both generation via differentiation and multiplication cant follow the path rho[v_i]->dj rho[v_i]->dj^2 rho[v_i]
# most probable fix is by considering indexed tensors of ranks >=2, but this bug isn't critical
def identify_equations(Q, reg_opts, library, threshold=1e-5, min_complexity=1,
                       max_complexity=None, max_equations=999, timed=True, excluded_terms=None):
    # trying to avoid retaining the values
    if excluded_terms is None:
        excluded_terms = set()
    if timed:
        start = timer()
    equations = []
    lambdas = []
    derived_eqns = {}
    # this can be eliminated by keeping track of two different max_complexities in args
    lib_max_complexity = max([term.complexity for term in library])  # generate list of derived terms up to here
    if max_complexity is None:
        max_complexity = int(np.ceil(lib_max_complexity))
    lib_prim_data = set([tup for term in library for tup in unpack_prims(term)])
    lib_prim_terms = make_terms(lib_prim_data)
    # print(lib_prim_terms)
    for complexity in range(min_complexity, max_complexity + 1):
        while len(equations) < max_equations:
            selection = [(term, i) for (i, term) in enumerate(library) if term.complexity <= complexity
                         and term not in excluded_terms]
            if len(selection) == 0:
                break
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
            for lhs, rhs in infer_equations(eq, lib_prim_terms, lib_max_complexity):
                excluded_terms.add(lhs)
                derived_eqns[str(eq)].append(form_equation(lhs, rhs))
    return equations, lambdas, derived_eqns, excluded_terms


def interleave_identify(Qs, reg_opts_list, libraries, threshold=1e-5, min_complexity=1,
                        max_complexity=None, max_equations=999, timed=True, excluded_terms=None):
    # trying to avoid retaining the values
    if excluded_terms is None:
        excluded_terms = set()
    equations = []
    lambdas = []
    derived_eqns = {}
    if max_complexity is None:
        max_complexity = int(np.ceil(max([term.complexity for library in libraries for term in library])))
    for complexity in range(min_complexity, max_complexity + 1):
        for Q, reg_opts, library in zip(Qs, reg_opts_list, libraries):
            eqs_i, lbds_i, der_eqns_i, exc_terms_i = identify_equations(Q, reg_opts, library,
                                                                        threshold=threshold, min_complexity=complexity,
                                                                        max_complexity=complexity,
                                                                        max_equations=max_equations, timed=timed,
                                                                        excluded_terms=excluded_terms)
            equations += eqs_i
            lambdas += lbds_i
            derived_eqns.update(der_eqns_i)
            excluded_terms.update(exc_terms_i)
    return equations, lambdas, derived_eqns, excluded_terms


def form_equation(lhs, rhs):
    if rhs is None:
        return Equation([lhs], [1])
    else:
        return TermSum([lhs]) + (-1) * rhs


def make_equation_from_Xi(Xi, lambd, best_term, lambda1, sublibrary):
    # print(Xi, lambd, best_term, lambda1, sublibrary)
    if lambda1 < lambd:
        return Equation([sublibrary[best_term]], [1]), lambda1
    else:
        zipped = [(sublibrary[i], c) for i, c in enumerate(Xi) if c != 0]
        return Equation([e[0] for e in zipped], [e[1] for e in zipped]), lambd


def unpack_prims(term):
    if isinstance(term, ConstantTerm):
        return (term, (), ()),
    else:
        # we desperately need to make this hashable
        return zip(term.obs_list, tuple(tuple(el) for el in term.der_index_list),
                   tuple(tuple(ell) for ell in term.obs_index_list))


def make_terms(lib_prim_data):
    lib_prim_terms = []
    for prim, der_ind, obs_ind in lib_prim_data:
        if isinstance(prim, ConstantTerm):
            lib_prim_terms.append(prim)
        else:
            tensor = LibraryTensor(prim)
            term = LibraryTerm(tensor, index_list=[list(der_ind), list(obs_ind)])
            lib_prim_terms.append(term)
    return lib_prim_terms


def infer_equations(equation, lib_prim_terms, max_complexity):
    lhs, rhs = equation.eliminate_complex_term()
    yield lhs, rhs
    if lhs.complexity >= max_complexity or isinstance(lhs, ConstantTerm):
        return
    # need to handle cases if lhs derivative has multiple terms and/or has coeff bigger than 1
    if rhs is not None:
        lhs_dt, rhs_dt = rebalance(lhs.dt(), rhs.dt())
        lhs_dx, rhs_dx = rebalance(lhs.dx(), rhs.dx())
        yield lhs_dt, rhs_dt
        yield lhs_dx, rhs_dx
        # compute multiplications
        for term in lib_prim_terms:
            if term.complexity + lhs.complexity <= max_complexity:
                yield term * lhs, term * rhs
    else:
        lhs_dt, rhs_dt = rebalance(lhs.dt(), None)
        lhs_dx, rhs_dx = rebalance(lhs.dx(), None)
        yield lhs_dt, rhs_dt
        yield lhs_dx, rhs_dx
        # compute multiplications
        for term in lib_prim_terms:
            if term.complexity + lhs.complexity <= max_complexity:
                yield term * lhs, None


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
