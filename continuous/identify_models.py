from library import *
from sparse_reg import *
from timeit import default_timer as timer

def identify_equations(Q, reg_opts, library, observables, threshold=1e-5, 
                       max_complexity=None, max_equations=999, timed=True, excluded_terms=[]):
    if timed:
        start = timer()
    obs_terms = [obs_to_term(obs) for obs in observables]
    equations = []
    lambdas = []
    derived_eqns = {}
    if max_complexity is None:
        allowed_terms = library
        max_complexity = max([term.complexity for term in library])
    for complexity in range(1, max_complexity+1):
        more_models = True
        while more_models:
            selection = [(term, i) for (i, term) in enumerate(library) if term.complexity<=complexity
                        and term not in excluded_terms]
            sublibrary = [s[0] for s in selection]
            inds = [s[1] for s in selection]
            reg_opts['subinds'] = inds
            #subQ = Q[:, inds]
            ### identify model
            eq, res = make_equation_from_Xi(*sparse_reg(
                                            Q, opts=reg_opts), sublibrary)
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
            for lhs, rhs in infer_equations(eq, obs_terms, max_complexity):
                excluded_terms.append(lhs)
                if rhs is None:
                    derived_eqns[str(eq)].append(Equation([lhs], [1]))
                else:
                    derived_eqns[str(eq)].append(-1*TermSum([lhs])+rhs)
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