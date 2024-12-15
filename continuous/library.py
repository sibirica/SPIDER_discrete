import copy
from functools import reduce
from numbers import Real
from operator import add
from typing import Any, Optional
from warnings import warn

#from numpy import inf

from commons.z3base import *
from commons.library import *

#def raw_library_tensors(observables: Iterable[Observable],
def valid_prime_lists(observables: Iterable[Observable],
                        obs_orders: List[int],
                        nt: int,
                        nx: int,
                        #rank: int,
                        max_order: DerivativeOrder | INF = INF(),
                        zeroidx: int = 0) -> Generator[Union[LibraryTerm, ConstantTerm], None, None]:
    """
    Instantiates a Generator that will yield all LibraryTerms using the inputs.
    NOTE: For exclusive use of generate_terms_to.

    :param observables: List of Observable object to generate tensors from.
    :param obs_orders: List of integers representing the number of appearances of each Observable.
    :param nt: Number of times a time derivative occurs.
    :param nx: Number of times a spatial derivative occurs.
    :param max_order: Maximum derivative order of a tensor.
    :param zeroidx: Index from which start to looking for non-zero terms in obs_order.
    :return: Generator that yields all LibraryTensor objects that can be constructed from the function's arguments.
    """
    # print(obs_orders, nt, nx, max_order)
    while obs_orders[zeroidx] == 0:  # find first observable with nonzero # of appearances
        zeroidx += 1
        if zeroidx == len(observables):  # if all indices are zero, yield ConstantTerm
            return
    if sum(obs_orders) == 1:  # checks for terms with only one observable
        i = obs_orders.index(1)
        do = DerivativeOrder.blank_derivative(nt, nx)
        if max_order >= do:  # only yields canonical tensor if derivative orders < max_order
            prim = LibraryPrime(derivative=do, derivand=observables[i])
            yield (prim,)
        return
    for i in range(nt + 1):
        for j in range(nx + 1):
            do = DerivativeOrder.blank_derivative(torder=i, xorder=j)
            if max_order >= do:  # only yields canonical tensor if its derivative orders < max_order
                prim = LibraryPrime(derivative=do, derivand=observables[zeroidx])
                new_orders = list(obs_orders)
                new_orders[zeroidx] -= 1
                if obs_orders[zeroidx] == 1:  # reset max_order since we are going to next observable
                    do = INF()
                for prim2 in valid_prime_lists(observables, new_orders, nt - i, nx - j, do, zeroidx):
                    yield prim2 + (prim,) 


def generate_terms_to(order: int,
                      observables: Iterable[Observable] = None,
                      max_rank: int = 2,
                      max_observables: int = 999) -> List[Union[ConstantTerm, LibraryTerm]]:
    """
    Given a list of Observable objects and a complexity order, returns the list of all LibraryTerms with complexity up to order and rank up to max_rank using at most max_observables copies of the observables.

    :param order: Max complexity order that terms will be generated to.
    :param observables: list of Observable objects used to construct the terms.
    :param max_observables: Maximum number of Observables (and derivatives) in a single term.
    :return: List of all possible LibraryTerms whose complexity is less than or equal to order, that can be generated
    using the given observables.
    """
    observables = sorted(observables, reverse=True) # make sure observables are in reverse order
    libterms = list()
    n = order  # max number of "blocks" to include
    k = len(observables)
    weights = [obs.complexity for obs in observables] + [1, 1] # complexities of each symbol
    for part in partition(n, k + 2, weights=weights):  # k observables + 2 derivative dimensions
        # not a valid term if no observables or max exceeded
        obs_orders = part[:k]
        if 0 < sum(obs_orders) <= max_observables:
            nt, nx = part[k:]
            for prime_list in valid_prime_lists(observables, obs_orders, nt, nx):
                parity = sum(len(prime.all_indices()) for prime in prime_list) % 2
                for rank in range(parity, max_rank + 1, 2):
                    term = LibraryTerm(primes=prime_list, rank=rank)
                    for labeled in generate_indexings(term):
                        # terms should already be in canonical form except eq_canon
                        libterms.append(labeled.eq_canon()[0])
    return libterms
