from __future__ import annotations

from dataclasses import dataclass, field, replace, KW_ONLY
from typing import List, Dict, Union, Tuple, Iterable, Generator
from itertools import permutations
from functools import cached_property
from collections import defaultdict, Counter

import numpy as np
from typing import Union

from z3_base import *

# increment all indices (Einstein or literal) in an expression
def inc_inds(expr: EinSumExpr[VarIndex | LiteralIndex]):
    return expr.map(expr_map=inc_inds, 
                    index_map=lambda ind: dataclasses.replace(ind, value=ind.value + 1))

# decrement all indices (Einstein or literal) in an expression
def dec_inds(expr: EinSumExpr[VarIndex | LiteralIndex]):
    return expr.map(expr_map=dec_inds, 
                    index_map=lambda ind: dataclasses.replace(ind, value=ind.value - 1))

def dt(expr: EinSumExpr[VarIndex]): 
    match expr:
        case DerivativeOrder():
            return DerivativeOrder(self.torder + 1, self.x_derivatives)
        case LibraryPrime():

        case LibraryTerm():

        case Equation():
        # note that derivative is an Equation object in general
            components = [coeff * term.dt() for term, coeff in zip(self.term_list, self.coeffs)
                          if not isinstance(term, ConstantTerm)]
            if not components:
                return Equation([], []) #None - might need an is_empty for Equation?
            return reduce(add, components)#.canonicalize()

# construct term or equation by taking x derivative with respect to new i index, shifting others up by 1
# NOT GUARANTEED TO BE CANONICAL
def dx(expr: EinSumExpr[VarIndex]): 
    dx_expr = dx_helper(inced)

# take x derivative without worrying about indices
def dx_helper(expr: EinSumExpr[VarIndex]):
    match expr:
        case DerivativeOrder():
            return DerivativeOrder(expr.torder, tuple([VarIndex(0)]+expr.x_derivatives))
        case LibraryPrime():

        case LibraryTerm():

        case Equation():
            components = [coeff * term.dx_helper() for term, coeff in zip(self.term_list, self.coeffs)
                      if not isinstance(term, ConstantTerm)]
            if not components:
                return Equation([], [])
            return reduce(add, components)#.canonicalize()

# contract term or equation along i and j indices, setting j to i (if i<j) and moving others down by 1
# NOT GUARANTEED TO BE CANONICAL
def contract(expr: EinSumExpr[VarIndex], i: VarIndex, j: VarIndex):

@dataclass
class DerivativeOrder[T](EinSumExpr):
    """
    Object to store and manipulate derivative orders.
    """
    torder: int = 0
    x_derivatives: Tuple[T] = None
    is_commutative: bool = True

    @cached_property
    def complexity(self):
        return self.torder+self.xorder

    @cached_property
    def xorder(self):    
        return len(self.x_derivatives)

    @classmethod
    def blank_derivative(cls, torder, xorder):
    # make an abstract x derivative with given orders 
        x_derivatives = tuple([IndexHole()]*xorder)
        return DerivativeOrder[IndexHole](torder, x_derivatives)

    @classmethod
    def indexed_derivative(cls, torder, xorder):
        x_derivatives = tuple([VarIndex[i] for i in range(xorder)])
        return DerivativeOrder[VarIndex](torder, x_derivatives)

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
        return tstring + xstring[:-1] # get rid of the trailing space
        #return f'DerivativeOrder({self.torder}, {self.xorder})'

    def __lt__(self, other):
        if not isinstance(other, Observable):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return self.torder < other.torder if self.torder != other.torder \
            else self.x_derivatives < other.x_derivatives

    def __repr__(self):
        index_string = ''.join([repr(idx) for idx in self.all_indices()])
        return f"{self.string}" if index_string == "" else f"{self.string}_{index_string}"

    def sub_exprs(self) -> Iterable[T]:
        return []

    def own_indices(self) -> Iterable[T]:
        return tuple([IndexHole()]*self.get_rank()) if self.indices is None  \
               else self.indices

    def map[T2](self, *, 
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map 
            and (direct) child indices according to index_map"""
        return replace(self, x_derivatives=tuple([index_map(index) for index in self.own_indices()]))

@dataclass(frozen=True)
class Observable[T](EinSumExpr):
    """
    Data class object that stores a string representation of an observable as well as its rank. For documentation
    purposes, this class will always be refered to as 'Observable' (capitalized), unless stated otherwise. Furthermore,
    the term 'observable' usually does NOT refer to this class, but rather to a LibraryPrimitive or IndexedPrimitive
    object.
    """
    _: KW_ONLY
    string: str  # String representing the Observable.
    indices: Tuple[T] = None
    is_commutative: bool = False

    @cached_property
    def complexity(self):
        return self.get_rank()

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
        return tuple([IndexHole()]*self.get_rank()) if self.indices is None  \
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

# NOTE: HIGHER-RANK TERMS IN SUM NOT GUARANTEED TO BE CANONICAL - CANONICALIZE WHEN SAMPLING
class Equation[T, Derivand]:  # can represent equation (expression = 0) OR expression
    def __init__(self, term_list, coeffs):  # terms are LibraryTerms, coeffs are real numbers
        content = zip(term_list, coeffs)
        coeffs = defaultdict(0)
        for term, coeff in content:
            coeffs[term] += coeff
        # remove terms with 0 coefficient
        for term in content.keys():
            if coeffs[term] == 0:
                del coeffs[term]
        # note that sorting guarantees canonicalization in equation term order
        self.term_list = sorted(coeffs.keys())
        self.coeffs = [coeffs[term] for term in self.term_list]
        self.rank = term_list[0].rank
        self.complexity = sum([term.complexity for term in term_list])  # another choice is simply the number of terms

    def __add__(self, other):
        if isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise TypeError(f"Second argument {other}) is not an equation.")

    def __rmul__(self, other):
        if isinstance(other, EinSumExpr): # multiplication by term
            return Equation([(other * term).canonicalize() for term in self.term_list], self.coeffs)
        else:  # multiplication by number
            return Equation(self.term_list, [other * c for c in self.coeffs])

    #def __mul__(self, other):
    #    return self.__rmul__(other)

    def __repr__(self):
        repstr = [str(coeff) + ' * ' + str(term) + ' + ' for term, coeff in zip(self.term_list, self.coeffs)]
        return reduce(add, repstr)[:-3]

    def __str__(self):
        return self.__repr__() + " = 0"

    def __eq__(self, other):
        return self.term_list == other.term_list and self.coeffs == other.coeffs

    def contract(self, ind1=0, ind2=1):
        components = [coeff * term.contract(ind1, ind2) for term, coeff in zip(self.term_list, self.coeffs)]
        if not components:
            return [] #None
        return reduce(add, components)

    # should no longer be needed since canonicalization enforced in the constructor
    # def canonicalize(self):
    #     if len(self.term_list) == 0:
    #         return self
    #     term_list = []
    #     coeffs = []
    #     i = 0
    #     while i < len(self.term_list):
    #         reps = 0
    #         prev = self.term_list[i]
    #         while i < len(self.term_list) and prev == self.term_list[i]:
    #             reps += self.coeffs[i]
    #             i += 1
    #         term_list.append(prev)
    #         coeffs.append(reps)
    #     return Equation(term_list, coeffs)

    def eliminate_complex_term(self, return_normalization=False):
        if len(self.term_list) == 1:
            return self.term_list[0], None
        lhs = max(self.term_list, key=lambda t: t.complexity)
        lhs_ind = self.term_list.index(lhs)
        new_term_list = self.term_list[:lhs_ind] + self.term_list[lhs_ind + 1:]
        new_coeffs = self.coeffs[:lhs_ind] + self.coeffs[lhs_ind + 1:]
        new_coeffs = [-c / self.coeffs[lhs_ind] for c in new_coeffs]
        rhs = Equation(new_term_list, new_coeffs)
        if return_normalization:
            return lhs, rhs, self.coeffs[lhs_ind]
        return lhs, rhs

    def to_term(self):
        if len(self.term_list) != 1:
            raise ValueError("Equation contains more than one distinct term")
        else:
            return self.term_list[0]

class TermSum(Equation):
    def __init__(self, term_list):  # terms are LibraryTerms, coeffs are real numbers
        self.term_list = sorted(term_list)
        self.coeffs = [1] * len(term_list)
        self.rank = term_list[0].rank

    def __str__(self):
        repstr = [str(term) + ' + ' for term in self.term_list]
        return reduce(add, repstr)[:-3]

    def __add__(self, other):
        #if isinstance(other, TermSum):
        #    return TermSum(self.term_list + other.term_list)
        #elif isinstance(other, Equation):
        if isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise TypeError(f"Second argument {other}) is not an equation.")

### utilities
# def create_derivative_string(torder: int, xorder: int) -> (str, str):
#     """
#     Creates a derivative string given a temporal order and a spatial order.

#     :param torder: Temporal derivative order.
#     :param xorder: Spatial Derivative Order.
#     :return: Time derivative string, Spatial derivative string
#     """

#     if torder == 0:
#         tstring = ""
#     elif torder == 1:
#         tstring = "dt "
#     else:
#         tstring = f"dt^{torder} "
#     if xorder == 0:
#         xstring = ""
#     elif xorder == 1:
#         xstring = "dx "
#     else:
#         xstring = f"dx^{xorder} "
#     return tstring, xstring

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
