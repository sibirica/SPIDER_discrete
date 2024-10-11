# It may or may not be nicer to take the SRDataset object as input for some of these
from timeit import default_timer as timer
from functools import reduce
from operators import add

from library import *
from commons.sparse_reg import *
from commons.sparse_reg_bf import *
    
def identify_equations(Q, reg_opts, library, print_opts=None, threshold=1e-5, min_complexity=1, # ranks=None,
                       max_complexity=None, max_equations=999, timed=True, excluded_terms=None, primes=None):
    if timed:
        start = timer()
    equations = []
    lambdas = []
    derived_eqns = {}
    #if ranks is None:
    #    ranks = (0, 1, 2)
    if print_opts is None:
        #print_opts = {sigfigs: 3, latex_output: False}
        print_opts = {num_format: '{0:.3g}', latex_output: False}
    if excluded_terms is None:
        excluded_terms = set()
    # this can be eliminated by keeping track of two different max_complexities in args
    lib_max_complexity = max([term.complexity for term in library])  # generate list of derived terms up to here
    if max_complexity is None:
        max_complexity = lib_max_complexity
    if primes is None:
        primes = get_primes(library, max_complexity)
    for complexity in range(min_complexity, max_complexity + 1):
        while len(equations) < max_equations:
            selection = [(term, i) for (i, term) in enumerate(library) if term.complexity <= complexity
                         and term not in excluded_terms]
            if len(selection) == 0:  # no valid terms of this complexity
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
            print(f'Identified model: {eq.pstr(print_opts)} (order {complexity}, residual {res:.2e})')
            # eliminate terms via infer_equations
            derived_eqns[str(eq)] = []
            for new_eq in infer_equations(eq, multipliers, lib_max_complexity):
                lhs, rhs = new_eq.eliminate_complex_term()
                # print(lhs)
                excluded_terms.add(lhs)
                derived_eqns[str(eq)].append(form_equation(lhs, rhs))
    return equations, lambdas, derived_eqns, excluded_terms


def interleave_identify(Qs, reg_opts_list, libraries, print_opts=None, threshold=1e-5, min_complexity=1,  # ranks = None
                        max_complexity=None, max_equations=999, timed=True, excluded_terms=None):
    equations = []
    lambdas = []
    derived_eqns = {}
    #if ranks is None:
    #    ranks = (0, 1, 2)
    if excluded_terms is None:
        excluded_terms = set()
    lib_max_complexity = max([term.complexity for library in libraries for term in library])
    if max_complexity is None:
        max_complexity = lib_max_complexity
    concat_libs = reduce(add, libraries, initial=[])
    primes = get_primes(concat_libs, max_complexity)
    for complexity in range(min_complexity, max_complexity + 1):
        for Q, reg_opts, library in zip(Qs, reg_opts_list, libraries):
            eqs_i, lbds_i, der_eqns_i, exc_terms_i = identify_equations(Q, reg_opts, library,
                                                                        threshold=threshold,
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
        return Equation(terms=(sublibrary[best_term],), coeffs=(1,)), lambda1
    else:
        zipped = [(sublibrary[i], c) for i, c in enumerate(Xi) if c != 0]
        return Equation(terms=[e[0] for e in zipped], coeffs=[e[1] for e in zipped]), lambd

def infer_equations(equation, primes, max_complexity, complexity=None):
    #yield equation
    # do all of the contractions in one step so we don't have different permutations of contraction & index creation 
    if complexity is None:
        complexity = max([term.complexity for term in equation.terms])
    if complexity >= max_complexity:
        return
    yield from get_all_contractions(equation)
    rem_complexity = max_complexity - complexity
    eq_dt = dt(equation)#.canonicalize() # I don't think canonicalization is necessary here
    eq_dx = dx(equation).canonicalize()
    yield from infer_equations(eq_dt, primes, max_complexity, complexity=complexity+1)
    yield from infer_equations(eq_dx, primes, max_complexity, complexity=complexity+1)
    for prime in primes:
        #if prime.complexity <= rem_complexity:
        # multiplication canonicalizes
        yield from infer_equations(prime * equation, primes, max_complexity, complexity=complexity+prime.complexity) 

def form_equation(lhs, rhs):
    if rhs is None:
        return Equation(terms=lhs, coeffs=(1,))
    else:
        return Equation(terms=lhs, coeffs=(1,)*len(lhs)) + (-1) * rhs

def get_all_contractions(equation):
    yield canonicalize(equation) # base case
    for i in range(equation.rank):
        for j in range(start=i+1, stop=equation.rank):
            yield from get_all_contractions(contract(equation, i, j))

def get_primes(library, max_complexity):
    all_primes = set(prime.purge_indices() for term in library 
                     for prime in term.primes if prime.complexity<=max_complexity)
    return all_primes