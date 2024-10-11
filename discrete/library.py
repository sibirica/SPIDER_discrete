import copy
from functools import reduce
from operator import add
from itertools import permutations
from numpy import prod
import numpy as np

from commons.z3base import *
from commons.library import *

@dataclass(frozen=True, order=True)
class CoarseGrainedProduct[T](EinSumExpr):
    """
    Dataclass representing rho[product]
    """
    _: KW_ONLY
    observables: Tuple[Observable]

    @cached_property
    def complexity(self):
        return 1+sum((observable.complexity) for observable in self.observables)
    pass

    @cached_property
    def rank(self): # only defined for VarIndex at the moment
        return index_rank(self.all_indices())

    def __repr__(self):
        return f"ρ[{' · '.join([repr(obs) for obs in self.observables])}]" \
               if len(self.observables)>0 else "ρ"

    def sub_exprs(self) -> Iterable[T]:
        return self.observables

    def own_indices(self) -> Iterable[T]:
        return ()

    def map[T2](self, *,
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map
            and (direct) child indices according to index_map"""
        return replace(self, observables=tuple(expr_map(obs) for obs in self.observables))

    def eq_canon(self):
        ecs = [obs.eq_canon() for obs in self.observables]
        sign = prod([pair[1] for pair in ecs], initial=1)
        return CoarseGrainedProduct(observables=tuple(sorted([pair[0] for pair in ecs]))), sign

def generate_terms_to(order: int,
                      observables: List[Observable],
                      max_rank: int = 2,
                      max_observables: int = 999,
                      max_rho: int = 999) -> List[Union[ConstantTerm, LibraryTerm]]:
    """
    Given a list of Observable objects and a complexity order, returns the list of all LibraryTerms with complexity up to order and rank up to max_rank using at most max_observables copies of the observables.

    :param order: Max complexity order that terms will be generated to.
    :param observables: list of Observable objects used to construct the terms.
    :param max_observables: Maximum number of Observables (and derivatives) in a single term.
    :return: List of all possible LibraryTerms whose complexity is less than or equal to order, that can be generated
    using the given observables.
    """
    #observables = sorted(observables, reverse=True) # make sure observables are in reverse order
    libterms = list()
    #libterms.append(ConstantTerm()) # ConstantTerm doesn't exist in discrete language
    n = order  # max number of "blocks" to include
    k = len(observables)
    partitions = [] # to make sure we don't duplicate partitions
    weights = [obs.complexity for obs in observables] + [1, 1] # complexities of each symbol
    # generate partitions in bijection to all possible primes
    for part in partition(n - 1, k + 2, weights=weights):  # k observables + 2 derivative dimensions, plus always 1 rho
        # account for complexities > 1
        #part = tuple([1]+list(part)) # ordering: [rho | observables | nt, nx]
        if sum(part[:k]) <= max_observables:
            # if part in partitions:
            #     continue
            # else:
            partitions.append(part)
            # print(part)

    def partition_to_prime(partition):
        prime_observables = []
        for i in range(k):
            prime_observables += [observables[i]] * partition[i]
        cgp = CoarseGrainedProduct(observables=tuple(prime_observables))
        derivative = DerivativeOrder.blank_derivative(torder=partition[-2], xorder=partition[-1])
        prime = LibraryPrime(derivative=derivative, derivand=cgp)
        return prime
    
    partitions = sorted(partitions)
    primes = [partition_to_prime(partition) for partition in partitions]
    #for pa, pr in zip(partitions, primes):
    #    print(pa, pr)

    # make all possible lists of primes and convert to terms of each rank, then generate labelings
    for prime_list in valid_prime_lists(primes, order, max_observables, max_rho):
        #print("PL:", prime_list)
        parity = sum(len(prime.all_indices()) for prime in prime_list) % 2
        #print("Parity:", parity)
        for rank in range(parity, max_rank + 1, 2):
            term = LibraryTerm(primes=prime_list, rank=rank)
            for labeled in generate_indexings(term):
                #canon = labeled.canonicalize()
                #libterms.append(canon)
                libterms.append(labeled) # terms should already be in canonical form  
    return libterms

def valid_prime_lists(primes: List[LibraryPrime],
                      order: int,
                      max_observables: int,
                      max_rho: int,
                      non_empty: bool = False) -> List[Union[ConstantTerm, LibraryTerm]]:
    # starting_ind: int
    """
    Generate components of valid terms from list of primes, with maximum complexity = order, maximum number of observables = max_observables, max number of primes = max_rho.
    """
    # , and using only primes starting from index starting_ind.
    # base case: yield no primes
    if non_empty:
        yield ()
    for i, prime in enumerate(primes): # relative_i
        #absolute_i = relative_i + starting_ind
        complexity = prime.complexity
        n_observables = len(prime.derivand.observables)
        #print(prime, complexity, n_observables)
        if complexity <= order and n_observables <= max_observables and 1 <= max_rho:
            for tail in valid_prime_lists(primes=primes[i:], order=order-complexity,
                                          max_observables=max_observables-n_observables, max_rho=max_rho-1,
                                          non_empty=True):
                yield (prime,) + tail