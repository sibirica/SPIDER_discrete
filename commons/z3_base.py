from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Union, assert_type
from abc import abstractmethod, ABC
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, KW_ONLY
from itertools import count
import z3

@dataclass
class SymmetryRep:
    _: KW_ONLY
    rank: int

    def __add__(self, other):
        print("Warning, SymmetryReps have been added")
        return FullRank(rank=self.rank + other.rank)

@dataclass
class Antisymmetric(SymmetryRep):
    def __repr__(self):
        return f"Antisymmetric rank {self.rank}"

@dataclass
class SymmetricTraceFree(SymmetryRep):
    def __repr__(self):
        return f"Symmetric trace-free rank {self.rank}"

@dataclass
class FullRank(SymmetryRep):
    def __repr__(self):
        return f"Rank {self.rank}"

Irrep = Antisymmetric | SymmetricTraceFree | FullRank

@dataclass(frozen=True)
class IndexHole:
    #id: int = field(default_factory=lambda counter=count(): next(counter))
    def __lt__(self, other):
        if not isinstance(other, IndexHole):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, IndexHole):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return True
    
    def __repr__(self):
        return "{ }"

@dataclass(frozen=True)
class SMTIndex:
    var: z3.ArithRef

    def __lt__(self, other):
        raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return None
    
    def __eq__(self, other):
        if not isinstance(other, SMTIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.var.eq(other.var)
            
    def __repr__(self):
        return f"{repr(self.var)}"

@dataclass(frozen=True)
class VarIndex:
    value: int

    def __lt__(self, other):
        if not isinstance(other, VarIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.value < other.value
    
    def __eq__(self, other):
        if not isinstance(other, VarIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.value == other.value

    def __repr__(self):
        return "αβγδεζηθικλμνξοπρστυφχψω"[self.value]

@dataclass(frozen=True)
class LiteralIndex:
    value: int

    def __lt__(self, other):
        if not isinstance(other, LiteralIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.value < other.value
    
    def __eq__(self, other):
        if not isinstance(other, LiteralIndex):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        else:
            return self.value == other.value
    
    def __repr__(self):
        return "xyzt"[self.value]  # if your problem has 5 dimensions you messed up     

Index = IndexHole | SMTIndex | VarIndex | LiteralIndex

@dataclass(frozen=True)
class EinSumExpr[T](ABC):
    _: KW_ONLY
    is_commutative: bool

    @abstractmethod
    def __lt__(self, other):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    # may need separate struct_eq if we need to manually check for terms commuting across *

    @abstractmethod
    def __repr__(self):
        ...

    def get_rank(self):
        match self.rank:
            case SymmetryRep(rank=rank):
                return rank
            case _ as rank:
                return rank

    @abstractmethod
    def sub_exprs(self) -> Iterable[EinSumExpr[T]]:
        """ Implementation returns list of sub_exprs (whatever this attribute may be called) """
        ...

    @abstractmethod
    def own_indices(self) -> Iterable[T]:
        """ Implementation returns list of own indices """
        ...

    #@lru_cache(maxsize=10000)
    def all_indices(self) -> list[T]: # make sure these are in depth-first/left-to-right order
        """ List all indices """
        return list(self.own_indices()) + [idx for expr in self.sub_exprs() for idx in expr.all_indices()]
        
    # def map(self, f, ctx):
    #     new_obj = {x: f(v, ctx) for x, v in self.__dict__.items()}
    #     for x, v in self.__dict__.items():
    #         if type(v) != type(new_obj[x]):
    #             raise TypeError(f"{new_obj[x]} is not a plausible replacement for {v} in {str(self.__class__)}.{x}")
    #     return dataclasses.replace(self, **new_obj)
            
    @abstractmethod
    def map[T2](self, *, 
                expr_map: Callable[[EinSumExpr[T]], EinSumExpr[T2]] = lambda x: x,
                index_map: Callable[[T], T2] = lambda x: x) -> EinSumExpr[T2]:
        """ Constructs a copy of self replacing (direct) child expressions according to expr_map 
            and (direct) child indices according to index_map"""
        ...

    def map_all_indices[T2](self, index_map: Callable[[T], T2]) -> EinSumExpr[T2]:
        def mapper(expr):
            nonlocal index_map
            return expr.map(expr_map=mapper, index_map=index_map)
        return mapper(self)

    def canonical_indexing_problem(self) -> tuple[EinSumExpr[SMTIndex], list[z3.ExprRef]]:        
        constraints = []
        def emap(expr):
            nonlocal constraints
            updated, cxs = expr.canonical_indexing_problem()
            constraints += cxs
            return updated

        base_id = f"i{id(self)}"
        def next_z3_var(ctr = count()):
            return z3.Int(f"{base_id}_{next(ctr)}")
        idx_cache = defaultdict(next_z3_var)
        def imap(idx):
            return SMTIndex(idx_cache[idx] if not isinstance(idx, IndexHole) 
                            else next_z3_var())
        
        updated = self.map(expr_map=emap, index_map=imap)
        
        if self.is_commutative:
            duplicates = defaultdict(list)
            for e, e_new in zip(self.sub_exprs(), updated.sub_exprs()):
                duplicates[e].append(e_new)
            for dup_list in duplicates.values():            
                for e, e_next in zip(dup_list, dup_list[1:]):
                    constraints.append(lexico_lt(e.all_indices(), e_next.all_indices()))
                    
        return updated, constraints

def generate_indexings(expr: EinSumExpr[IndexHole | VarIndex]) -> Iterable[EinSumExpr[VarIndex]]:
    indexed_expr, constraints = expr.canonical_indexing_problem() # includes lexicographic constraints    
    assert_type(indexed_expr, EinSumExpr[SMTIndex])
    # add global constraints
    indices = indexed_expr.all_indices()
    n_single_inds = expr.rank
    n_total_inds = (len(indices)+n_single_inds)//2
    # use-next-variable constraints
    single_idx_max = 0
    paired_idx_max = n_single_inds
    for j, idx in enumerate(indices):
        s_idx_max_next = z3.Int(f's_idxmax_{j}')
        p_idx_max_next = z3.Int(f'p_idxmax_{j}')
        constraints += [z3.Or(
            z3.And(idx.var == single_idx_max,
                   s_idx_max_next == idx.var+1, p_idx_max_next == paired_idx_max),
            z3.And(idx.var >= n_single_inds, idx.var <= paired_idx_max,
                   s_idx_max_next == single_idx_max, 
                   p_idx_max_next == paired_idx_max + z3.If(idx.var==paired_idx_max, 1, 0)
                  )
        )]
        single_idx_max = s_idx_max_next
        paired_idx_max = p_idx_max_next
    constraints += [single_idx_max == n_single_inds, paired_idx_max == n_total_inds]
    # constrain number of appearances in pair
    for paired_idx in range(n_single_inds, n_total_inds):
        constraints.append(z3.AtMost(*[idx == paired_idx for idx in indices], 2))
    # give problem to smt solver
    solver = z3.Solver()
    solver.add(*constraints)
    while (result := solver.check()) == z3.sat: # smt solver finds a new solution
        m = solver.model()
        indexing = {index: m[index.var] for index in indices}
        yield indexed_expr.map_all_indices(index_map = lambda index: VarIndex(indexing[index].as_long()))
        # prevent smt solver from repeating solution
        solver.add(z3.Or(*[idx.var != val for idx, val in indexing.items()])) 
    if result == z3.unknown:
        raise RuntimeError("Could not solve SMT problem :(")

def lexico_lt(idsA: list[z3.Int], idsB: list[z3.Int]) -> z3.ExprRef:
    lt = False
    for a, b in reversed(zip(idsA, idsB)):
        lt = z3.Or(a < b, z3.And(a == b, lt))
    return lt