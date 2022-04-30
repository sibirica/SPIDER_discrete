from library import *
from sparse_reg import *
from timeit import default_timer as timer

# does not properly identify all implications since generation via dx is incomplete from an indexing perspective
# e.g., both generation via differentiation and multiplication cannot follow the path rho[v_i]->dj rho[v_i]->dj^2 rho[v_i]
# most probable fix is by considering indexed tensors of ranks >=2, but this bug isn't critical
def identify_equations(Q, reg_opts, library, observables, threshold=1e-5, min_complexity=1,
                       max_complexity=None, max_equations=999, timed=True, excluded_terms=set()):
    if timed:
        start = timer()
    obs_terms = [obs_to_term(obs) for obs in observables]
    equations = []
    lambdas = []
    derived_eqns = {}
    # this can be eliminated by keeping track of two different max_complexities in args
    lib_max_complexity = max([term.complexity for term in library]) # generate list of derived terms up to here
    if max_complexity is None:
        max_complexity = lib_max_complexity
    for complexity in range(min_complexity, max_complexity+1):
        while len(equations)<max_equations:
            selection = [(term, i) for (i, term) in enumerate(library) if term.complexity<=complexity
                        and term not in excluded_terms]
            sublibrary = [s[0] for s in selection]
            inds = [s[1] for s in selection]
            reg_opts['subinds'] = inds
            ### identify model
            eq, res = make_equation_from_Xi(*sparse_reg(
                                            Q, **reg_opts), sublibrary)
            if res > threshold:
                break
            equations.append(eq)
            lambdas.append(res)
            ### add some output about the discovered model
            if timed:
                time = timer()-start
                print(f"[{time:.2f} s]")
            print(f'Identified model: {eq} (order {complexity}, residual {res:.2e})')
            ### eliminate terms via infer_equations
            derived_eqns[str(eq)] = []
            for lhs, rhs in infer_equations(eq, obs_terms, lib_max_complexity):
                excluded_terms.add(lhs)
                if rhs is None:
                    derived_eqns[str(eq)].append(Equation([lhs], [1]))
                else:
                    derived_eqns[str(eq)].append(-1*TermSum([lhs])+rhs)
    return equations, lambdas, derived_eqns, excluded_terms

def interleave_identify(Qs, reg_opts_list, libraries, observables, threshold=1e-5, min_complexity=1,
                        max_complexity=None, max_equations=999, timed=True, excluded_terms=set()):
    equations = []
    lambdas = []
    derived_eqns = {}
    if max_complexity is None:
        max_complexity = max([term.complexity for library in libraries for term in library]) 
    for complexity in range(min_complexity, max_complexity+1):
        for Q, reg_opts, library in zip(Qs, reg_opts_list, libraries):
            eqs_i, lbds_i, der_eqns_i, exc_terms_i = identify_equations(Q, reg_opts, library,
                    observables, threshold=threshold, min_complexity=complexity, max_complexity=complexity,
                    max_equations=max_equations, timed=timed, excluded_terms=excluded_terms)
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
    
def infer_equations(equation, obs_terms, max_complexity):
    lhs, rhs = equation.eliminate_complex_term()
    yield lhs, rhs
    if lhs.complexity >= max_complexity: # should be at most equal actually
        return
    # need to handle cases if lhs derivative has multiple terms and/or has coeff bigger than 1
    if rhs is not None:
        lhs_dt, rhs_dt = rebalance(lhs.dt(), rhs.dt())
        lhs_dx, rhs_dx = rebalance(lhs.dx(), rhs.dx())
        yield lhs_dt, rhs_dt
        yield lhs_dx, rhs_dx
        # compute multiplications
        for term in obs_terms:
            yield term*lhs, term*rhs
    else:
        lhs_dt, rhs_dt = rebalance(lhs.dt(), None)
        lhs_dx, rhs_dx = rebalance(lhs.dx(), None)
        yield lhs_dt, rhs_dt
        yield lhs_dx, rhs_dx
        # compute multiplications
        for term in obs_terms:
            yield term*lhs, None

def obs_to_term(observable):
    prim = LibraryPrimitive(DerivativeOrder(0, 0), observable)
    tensor = LibraryTensor(prim)
    labels = {k: [1] for k in list(range(tensor.rank))}
    return LibraryTerm(tensor, labels=labels)

def rebalance(lhs, rhs):
    if len(lhs.term_list) > 1: # note that we don't need to check =0 since 1 can't be the most complex term
        lhs1, lhs2, nrm = lhs.eliminate_complex_term(return_normalization=True)
        if rhs is None:
            new_rhs = -1*lhs2
        else:
            new_rhs = (1/nrm)*rhs + (-1*lhs2)
        return lhs1, new_rhs
    else:
        return lhs.to_term(), rhs