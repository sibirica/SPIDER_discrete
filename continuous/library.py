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
    Instatiates a Generator that will yield all LibraryTerms using the inputs.
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
            #yield ConstantTerm()
            return
    if sum(obs_orders) == 1:  # checks for terms with only one observable
        i = obs_orders.index(1)
        do = DerivativeOrder.blank_derivative(nt, nx)
        if max_order >= do:  # only yields canonical tensor if derivative orders < max_order
            prim = LibraryPrime(derivative=do, derivand=observables[i])
            yield (prim,)
            #yield LibraryTerm(primitives=(prim,), rank=rank) # this determines the rank of the product
        return
    for i in range(nt + 1):
        for j in range(nx + 1):
            do = DerivativeOrder.blank_derivative(torder=i, xorder=j)
            if max_order >= do:  # only yields canonical tensor if its derivative orders < max_order
                prim = LibraryPrime(derivative=do, derivand=observables[zeroidx])
                #term1 = LibraryTerm(primes=(prim,), rank=0)
                new_orders = list(obs_orders)
                new_orders[zeroidx] -= 1
                if obs_orders[zeroidx] == 1:  # reset max_order since we are going to next observable
                    max_order = INF()
                #for term2 in raw_library_tensors(observables, new_orders, nt - i, nx - j, rank, do, zeroidx):
                for prim2 in valid_prime_lists(observables, new_orders, nt - i, nx - j, do, zeroidx):
                    yield prim2 + (prim,) 
                    #yield term2 * term1  # reverse back to ascending order here


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
    #libterms.append(ConstantTerm()) # I think this is already handled
    n = order  # max number of "blocks" to include
    k = len(observables)
    weights = [obs.complexity for obs in observables] + [1, 1] # complexities of each symbol
    #to_skip = 20
    for part in partition(n, k + 2, weights=weights):  # k observables + 2 derivative dimensions
        # # account for complexities > 1
        # for i in range(k):
        #     part[i] = part[i] // observables[i].complexity
        # if part in partitions:
        #     continue
        # else:
        #     partitions.add(part)
        #if to_skip > 0:
        #    to_skip -= 1
        #    continue
        #print(part)
        # not a valid term if no observables or max exceeded
        obs_orders = part[:k]
        if 0 < sum(obs_orders) <= max_observables:
            nt, nx = part[k:]
            #print("...", obs_orders, nt, nx)
            #for rank in range(max_rank + 1):
            #    for tensor in raw_library_tensors(observables, obs_orders, nt, nx, rank):
            for prime_list in valid_prime_lists(observables, obs_orders, nt, nx):
                parity = sum(len(prime.all_indices()) for prime in prime_list) % 2
                #print(prime_list, parity)
                for rank in range(parity, max_rank + 1, 2):
                    term = LibraryTerm(primes=prime_list, rank=rank)
                    #print("term:", term, "rank:", rank)
                    for labeled in generate_indexings(term):
                        #canon = labeled.canonicalize()
                        #libterms.append(canon)
                        #print(labeled)
                        libterms.append(labeled) # terms should already be in canonical form
    return libterms
