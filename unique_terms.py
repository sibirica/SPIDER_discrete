# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:42:48 2021

@author: akash
"""

from typing import List
from itertools import permutations, combinations


from dataclasses import dataclass, field
from typing import TypeVar, Union, Generator, Tuple

T = TypeVar('T')

IndexingScaffold = Union['Index', 'RepeatScaffold', 'ProductScaffold']


@dataclass(frozen=True)
class Index:
    value: int = -1
    capacity: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, 'capacity', 1 if (self.value < 0) else 0)

    def __str__(self):
        if self.value < 0 and self.blocked_size > 0:
            return "(!{})".format(self.blocked_size)
        return str(self.value) if self.value >= 0 else "?"


@dataclass(frozen=True)
class RepeatScaffold:
    template: IndexingScaffold
    n: int
    capacity: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, 'capacity', self.n * self.template.capacity)

    def __str__(self):
        return "({})^{}".format(self.template, self.n)


@dataclass(frozen=True)
class ProductScaffold:
    l: IndexingScaffold
    r: IndexingScaffold
    blocked_size: int = 0
    capacity: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, 'capacity', self.l.capacity + self.r.capacity)

    def __str__(self):
        return str(self.l) + " " + str(self.r)


def swappable_terms(s: IndexingScaffold, offset=0) -> List[Tuple[int, int, int]]:
    if isinstance(s, Index):
        return []
    elif isinstance(s, RepeatScaffold):
        template_swaps = swappable_terms(s.template)
        offsets = [offset + i * s.template.capacity for i in range(s.n)]
        swaps = [(sz, fst + o, snd + o)
                 for o in offsets
                 for (sz, fst, snd) in template_swaps]
        return swaps + [(s.template.capacity, fst, snd)
                        for fst, snd in combinations(offsets, 2)]
    elif isinstance(s, ProductScaffold):
        return (
            swappable_terms(s.l, offset)
            + swappable_terms(s.r, offset + s.l.capacity)
        )


def swap_indices(indexing: List[int], fst: int, snd: int) -> List[int]:
    return tuple(i if i != fst and i != snd else (snd if i == fst else fst)
                 for i in indexing)


def swap_terms(
        indexing: List[int], term_size: int, fst: int, snd: int
        ) -> List[int]:
    if fst > snd:
        fst, snd = snd, fst
    return (
        indexing[:fst] 
        + indexing[snd:(snd + term_size)]
        + indexing[(fst + term_size):snd]
        + indexing[fst:(fst + term_size)] 
        + indexing[(snd + term_size):]
    )


def unique_indexings(
        s: IndexingScaffold, indices: List[Tuple[int, int]]
        ) -> Generator[List[int], None, None]:

    indices_by_size = {}
    max_index = 0
    for n_repeats, n_of_index in indices:
        indices_by_size[n_repeats] = \
            list(range(max_index, max_index + n_of_index))
        max_index += n_of_index

    index_swaps = [(i, j) for inds in indices_by_size.values()
                   for i, j in combinations(inds, 2)]

    term_swaps = swappable_terms(s)

    seen = set()
    init = sum((inds * sz for sz, inds in indices_by_size.items()), start=[])
    for indexing in permutations(init):

        if indexing in seen:
            continue
        seen.add(indexing)

        yield indexing

        # Remove all equivalent indexings
        to_visit = [indexing]
        while to_visit:
            cur = to_visit.pop()
            equivs = (
                [swap_indices(cur, i, j) for i, j in index_swaps]
                + [swap_terms(cur, sz, start, end)
                   for sz, start, end in term_swaps]
            )
            # check for a relabeling of indices
            for e in equivs:
                if e in seen:
                    continue
                seen.add(e)
                to_visit.append(e)


if __name__ == "__main__":

    v = Index()
    dxv = ProductScaffold(Index(), Index())

    print("v (dx v) (dx v)")
    v_dxv_dxv = ProductScaffold(v, RepeatScaffold(dxv, 2))

    inds = unique_indexings(v_dxv_dxv, [(1, 1), (2, 2)])
    for i, indexing in enumerate(inds):
        print("{:3}. {}".format(i, indexing))

    print()

    dxdxrho = RepeatScaffold(Index(), 2)
    print("v v (dx dx rho) (dx dx rho) (dx dx rho)")
    v_v_dxdxr_dxdxr_dxdxr = ProductScaffold(
        RepeatScaffold(v, 2),
        RepeatScaffold(dxdxrho, 3)
    )

    inds = unique_indexings(v_v_dxdxr_dxdxr_dxdxr, [(2, 4)])
    for i, indexing in enumerate(inds):
        print("{:3}. {}".format(i, indexing))

    print()

    dxdxv = ProductScaffold(RepeatScaffold(Index(), 2), v)
    print("(dx dx v) v")
    dxdxvv = ProductScaffold(dxdxv, v)

    inds = unique_indexings(dxdxvv, [(2, 2)])
    for i, indexing in enumerate(inds):
        print("{:3}. {}".format(i, indexing))

    print()

    print("(dx dx rho) (dx v)")
    xxrxv = ProductScaffold(dxdxrho, ProductScaffold(Index(), Index()))

    inds = unique_indexings(xxrxv, [(2, 2)])
    for i, indexing in enumerate(inds):
        print("{:3}. {}".format(i, indexing))
