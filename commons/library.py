from __future__ import annotations

from dataclasses import dataclass, field, replace, KW_ONLY
from typing import List, Dict, Union, Tuple, Iterable, Generator
from itertools import permutations
from functools import cached_property, reduce
from operator import add
from collections import defaultdict, Counter

import numpy as np
from typing import Union

from z3_base import *

# increment all indices (Einstein or literal) in an expression
def inc_inds(expr: EinSumExpr[VarIndex | LiteralIndex], shift=1):
    return expr.map(expr_map=inc_inds, 
                    index_map=lambda ind: replace(ind, value=ind.value + shift))
    
# get rank of an expression by looking at the indices of an Einstein expression
def index_rank(indices: Iterable[VarIndex]):
    index_counter = Counter(indices)
    num_singles = len([count for index, count in index_counter.items() if count==1])
    return num_singles

# get highest index in list
def highest_index(indices: Iterable[VarIndex]):
    return max(index.value for index in indices)

def canonicalize(expr: EinSumExpr[VarIndex]):
    indexings = generate_indexings(expr)
    canon = next(indexings)
    assert next(indexings, -1) == -1, "Expected only one canonical indexing" 
    return canon

def dt(expr: EinSumExpr[VarIndex]): 
    match expr:
        case Observable():
            return LibraryPrime(derivative=DerivativeOrder.indexed_derivative(1, 0), derivand=expr)
        case DerivativeOrder():
            return DerivativeOrder(torder=self.torder + 1, xorder=self.xorder, 
                                   x_derivatives=self.x_derivatives)
        case LibraryPrime():
            return replace(expr, derivative=dt(expr.derivative))
        case LibraryTerm():
            pass
        case Equation():
        # note that derivative is an Equation object in general
            components = tuple([coeff * dt(term) for term, coeff in zip(self.terms, self.coeffs)
                          if not isinstance(term, ConstantTerm)])
            if not components:
                return Equation((), ()) #None - might need an is_empty for Equation?
            return reduce(add, components)#.canonicalize()

# construct term or equation by taking x derivative with respect to new i index, shifting others up by 1
# NOT GUARANTEED TO BE CANONICAL
def dx(expr: EinSumExpr[VarIndex]): 
    # the alternative implementation was to run dx and then use z3 solver to identify index labeling
    inced = inc_inds(expr)
    return dx_helper(inced)

# take x derivative without worrying about indices
def dx_helper(expr: EinSumExpr[VarIndex]):
    match expr:
        case Observable():
            return LibraryPrime(derivative=DerivativeOrder.indexed_derivative(0, 1), derivand=expr)
        case DerivativeOrder(torder=to, x_derivatives=xd):
            return DerivativeOrder(torder=to, x_derivatives=(VarIndex(0), *xd))
        case LibraryPrime(derivative=d):
            return replace(expr, derivative=dx_helper(d))
        case LibraryTerm():
            subexs = list(expr.sub_exprs())
            return reduce(add, (reduce(mul_helper, (*subexs[:i], dx_helper(term), *subexs[i+1:])) 
                       for i, term in enumerate(subexs)))
        case Equation():
            components = tuple([coeff * dx_helper(term) for term, coeff in zip(self.terms, self.coeffs)
                      if not isinstance(term, ConstantTerm)])
            if not components:
                return Equation((), ())
            return reduce(add, components)#.canonicalize()

# contract term or equation along i and j indices, setting j to i (if i<j) and moving others down by 1
# NOT GUARANTEED TO BE CANONICAL
def contract(expr: EinSumExpr[VarIndex], i: int, j: int):
    if j<i:
        i, j = j, i
    def contraction_map(ind: VarIndex):
        if ind.value==j:
            return VarIndex(i)
        if ind.value>j:
            return VarIndex(ind.value-1)
        return ind
    return expr.map_all_indices(index_map=contraction_map).lexico_canon()

# helper function for prime/library term multiplication
def mul_helper(a: LibraryPrime | LibraryTerm, b: LibraryPrime | LibraryTerm):
    def cast_to_term(x):
        if isinstance(x, ConstantTerm):
            return LibraryTerm(primes=(), rank=0)
        return x if isinstance(x, LibraryTerm) else LibraryTerm(primes=(x,), rank=x.rank)
    a = cast_to_term(a)
    b = cast_to_term(b)
    shift = highest_index(a.all_indices()) + 1
    product_rank = a.rank + b.rank
    combined_primes = tuple(sorted([*(a.primes), *(inc_inds(b, shift).primes)]))
    product = LibraryTerm(primes=combined_primes, rank=product_rank)
    return canonicalize(product)

@dataclass(frozen=True)
class DerivativeOrder[T](EinSumExpr):
    """
    Object to store and manipulate derivative orders.
    """
    _: KW_ONLY
    torder: int = 0
    x_derivatives: Tuple[T]
    is_commutative: bool = True

    @cached_property
    def complexity(self):
        return self.torder+self.xorder

    @cached_property
    def xorder(self):    
        return len(self.x_derivatives)

    # @cached_property
    # def rank(self):
    #     match T:
    #         case IndexHole():
    #             return len(self.x_derivatives)
    #         case VarIndex():
    #             return index_rank(self)
    #         case _:
    #             raise NotImplementedError("")

    @classmethod
    def blank_derivative(cls, torder, xorder):
    # make an abstract x derivative with given orders 
        x_derivatives = tuple([IndexHole()]*xorder)
        return DerivativeOrder[IndexHole](torder=torder, x_derivatives=x_derivatives)

    @classmethod
    def indexed_derivative(cls, torder, xorder):
        x_derivatives = tuple([VarIndex(i) for i in range(xorder)])
        return DerivativeOrder[VarIndex](torder=torder, x_derivatives=x_derivatives)

    def __repr__(self):
        if self.torder == 0:
            tstring = ""
        elif self.torder == 1:
            tstring = "∂t "
        else:
            tstring = f"∂t^{self.torder} "
        xstring = ""
        if self.xorder != 0:
            ind_counter = Counter(self.x_derivatives)
            for ind in sorted(ind_counter.keys()):
                count = ind_counter[ind]
                if count == 1:
                    xstring += f"∂{ind} "
                else:
                    xstring += f"∂{ind}^{count} "
        return (tstring + xstring)[:-1] # get rid of the trailing space

    def __lt__(self, other):
        if not isinstance(other, DerivativeOrder):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return self.torder < other.torder if self.torder != other.torder \
            else self.x_derivatives < other.x_derivatives

    def sub_exprs(self) -> Iterable[T]:
        return ()

    def own_indices(self) -> Iterable[T]:
        #return tuple([IndexHole()]*self.rank) if self.x_derivatives is None  \
        #       else self.x_derivatives
        return self.x_derivatives

    def map[T2](self, *, 
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map 
            and (direct) child indices according to index_map"""
        return replace(self, x_derivatives=tuple([index_map(index) for index in self.own_indices()]))

@dataclass(frozen=True)
class Observable[T](EinSumExpr):
    """
    Data class object that stores a string representation of an observable as well as its rank. For
    documentation purposes, this class will always be refered to as 'Observable' (capitalized), unless
    stated otherwise.
    """
    _: KW_ONLY
    string: str  # String representing the Observable.
    rank: Union[int, SymmetryRep]
    indices: Tuple[T] = None
    is_commutative: bool = False

    @cached_property
    def complexity(self):
        return self.rank

    # For sorting: convention is in ascending order of name

    def __lt__(self, other):
        if not isinstance(other, Observable):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return self.string < other.string if self.string != other.string \
            else tuple(self.all_indices()) < tuple(other.all_indices())

    def __repr__(self):
        index_string = ''.join([repr(idx) for idx in self.all_indices()])
        return f"{self.string}" if index_string == "" else f"{self.string}_{index_string}"

    def sub_exprs(self) -> Iterable[T]:
        return []

    def own_indices(self) -> Iterable[T]:
        return tuple([IndexHole()]*self.rank) if self.indices is None  \
               else self.indices

    def map[T2](self, *, 
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map 
            and (direct) child indices according to index_map"""
        return replace(self, indices=tuple([index_map(index) for index in self.own_indices()]))

@dataclass(frozen=True)
class ConstantTerm(Observable):
    """ Short-hand for constant term = 1 """

    string: str = "1"
    rank: int = 0

    def __repr__(self):
        return "1"

    def dx(self):
        return 0

    def dt(self):
        return 0

    def __mul__(self, other):
        return other

    def __rmul__(self, other):
        return other

@dataclass(frozen=True)
class LibraryPrime[T, Derivand](EinSumExpr):
    """
    Dataclass representing DerivativeOrder applied to a Derivand (e.g. Observable, CGP)
    """
    _: KW_ONLY
    derivative: DerivativeOrder
    derivand: Derivand
    is_commutative: bool = False

    @cached_property
    def complexity(self):
        return self.derivative.complexity+self.derivand.complexity

    @cached_property
    def rank(self): # only defined for VarIndex at the moment
        return index_rank(self.all_indices())

    # For sorting: convention is in ascending order of name of derivand, then derivative

    def __lt__(self, other):
        if not isinstance(other, LibraryPrime):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return self.derivand < other.derivand if self.derivand != other.derivand \
            else self.derivative < other.derivative

    def __repr__(self):
        string1 = repr(self.derivative)
        string2 = repr(self.derivand)
        return f"{string1 + " " if string1 else ""}{string2}"

    def sub_exprs(self) -> Iterable[T]:
        return (self.derivative, self.derivand)

    def own_indices(self) -> Iterable[T]:
        return ()

    def map[T2](self, *, 
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map 
            and (direct) child indices according to index_map"""
        return replace(self, derivative=expr_map(self.derivative), derivand=expr_map(self.derivand))

    def __mul__(self, other: Union[LibraryPrime, LibraryTerm]) -> LibraryTerm:
        return mul_helper(self, other)

@dataclass(frozen=True, order=True)
class LibraryTerm[T, Derivand](EinSumExpr):
    """
    Dataclass representing DerivativeOrder applied to a Derivand (e.g. Observable, CGP)
    """
    _: KW_ONLY
    primes: Tuple[LibraryPrime[T, Derivand]]
    rank: Union[int, SymmetryRep]
    is_commutative: bool = True

    @cached_property
    def complexity(self):
        return sum((prime.complexity) for prime in self.primes)

    #@cached_property
    #def rank(self): # only defined for VarIndex at the moment
    #    return index_rank(self)

    # For sorting: convention is in ascending order of name

    # def __lt__(self, other):
    #     if not isinstance(other, Observable):
    #         raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
    #     return self. < other.string if self.string != other.string \
    #         else tuple(self.all_indices()) < tuple(other.all_indices())


    def __repr__(self):
        return ' · '.join([repr(prime) for prime in self.primes])

    def sub_exprs(self) -> Iterable[T]:
        return self.primes

    def own_indices(self) -> Iterable[T]:
        return ()

    def map[T2](self, *, 
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map 
            and (direct) child indices according to index_map"""
        return replace(self, primes=tuple(expr_map(prime) for prime in self.primes))

    def __add__(self, other): # add to LibraryTerm or Equation
        if isinstance(other, LibraryTerm):
            return Equation(coeffs=(1, 1), terms=(self, other))
        else:
            return Equation(coeffs=(1, *other.coeffs), terms=(self, *other.terms))

    def __mul__(self, other: Union[LibraryPrime, LibraryTerm]) -> LibraryTerm:
        return mul_helper(self, other)     

    def lexico_canon(self):
        return LibraryTerm(primes=tuple(sorted(self.primes)), rank=self.rank)

# NOTE: HIGHER-RANK TERMS IN SUM NOT GUARANTEED TO BE CANONICAL - CANONICALIZE WHEN SAMPLING
class Equation[T, Derivand]:  # can represent equation (expression = 0) OR expression
    def __init__(self, terms, coeffs):  # terms are LibraryTerms, coeffs are real numbers
        content = zip(terms, coeffs)
        coeffs = defaultdict(int)
        for term, coeff in content:
            coeffs[term] += coeff
        # remove terms with 0 coefficient
        coeffs = {term: coeff for term, coeff in coeffs.items() if coeff != 0}
        # note that sorting guarantees canonicalization in equation term order
        self.terms = tuple(sorted(coeffs.keys()))
        self.coeffs = tuple(coeffs[term] for term in self.terms)
        self.rank = terms[0].rank
        self.complexity = sum([term.complexity for term in terms])  # another choice is simply the number of terms

    def __add__(self, other):
        if isinstance(other, Equation):
            return Equation(self.terms + other.terms, self.coeffs + other.coeffs)
        else:
            return other + self
            #raise TypeError(f"Second argument {other}) is not an equation.")

    def __mul__(self, other):
        if isinstance(other, EinSumExpr): # multiplication by term
            # may need to canonicalize term * other - more likely not though
            return Equation([term * other for term in self.terms], self.coeffs)
        else:  # multiplication by number
            return Equation(self.terms, [c * other for c in self.coeffs])

    #def __rmul__(self, other):
    #    return other.__mul__(self)

    def __repr__(self):
        repstr = ' + '.join([str(coeff) + ' · ' + str(term) if coeff != 1 else str(term)
                             for coeff, term in zip(self.coeffs, self.terms)])
        return repstr

    def __str__(self):
        return self.__repr__() + " = 0"

    def __eq__(self, other):
        return self.terms == other.terms and self.coeffs == other.coeffs

    # should no longer be needed since canonicalization enforced in the constructor
    # def canonicalize(self):
    #     if len(self.terms) == 0:
    #         return self
    #     terms = []
    #     coeffs = []
    #     i = 0
    #     while i < len(self.terms):
    #         reps = 0
    #         prev = self.terms[i]
    #         while i < len(self.terms) and prev == self.terms[i]:
    #             reps += self.coeffs[i]
    #             i += 1
    #         terms.append(prev)
    #         coeffs.append(reps)
    #     return Equation(terms, coeffs)

    # note that LHS should be canonicalized if needed for lookup
    def eliminate_complex_term(self, return_normalization=False):
        if len(self.terms) == 1:
            return self.terms[0], None
        lhs = max(self.terms, key=lambda t: t.complexity)
        lhs_ind = self.terms.index(lhs)
        new_terms = self.terms[:lhs_ind] + self.terms[lhs_ind + 1:]
        new_coeffs = self.coeffs[:lhs_ind] + self.coeffs[lhs_ind + 1:]
        new_coeffs = [-c / self.coeffs[lhs_ind] for c in new_coeffs]
        rhs = Equation(new_terms, new_coeffs)
        if return_normalization:
            return lhs, rhs, self.coeffs[lhs_ind]
        return lhs, rhs

    def to_term(self):
        if len(self.terms) != 1:
            raise ValueError("Equation contains more than one distinct term")
        else:
            return canonicalize(self.terms[0]) # may need structural canonicalization too - check

# class TermSum(Equation):
#     def __init__(self, terms):  # terms are LibraryTerms, coeffs are real numbers
#         self.terms = sorted(terms)
#         self.coeffs = [1] * len(terms)
#         self.rank = terms[0].rank

#     def __repr__(self):
#         repstr = ' + '.join([str(term) for term in self.terms])
#         return repstr

#     def __add__(self, other):
#         #if isinstance(other, TermSum):
#         #    return TermSum(self.terms + other.terms)
#         #elif isinstance(other, Equation):
#         if isinstance(other, Equation):
#             return Equation(self.terms + other.terms, self.coeffs + other.coeffs)
#         else:
#             raise TypeError(f"Second argument {other}) is not an equation.")

def yield_tuples_up_to(bounds):
    if len(bounds) == 0:
        yield ()
        return
    for i in range(bounds[0] + 1):
        for tup in yield_tuples_up_to(bounds[1:]):
            # print(i, tup)
            yield (i,) + tup

def yield_legal_tuples(bounds): # allocate observables & derivatives up to the available bounds
    # print("bounds:", bounds)
    if sum(bounds[:-2]) > 0:  # if there are still other observables left
        # print("ORDERS:", bounds)
        yield from yield_tuples_up_to(bounds)
    else:  # must return all derivatives immediately
        # print("Dump ORDERS")
        yield bounds
