from functools import lru_cache
from typing import Protocol, assert_type
from abc import abstractmethod
from collections.abc import Callable
from dataclass import dataclass, field
from itertools import count
import z3

@dataclass(frozen=True)
class IndexHole:
    #id: int = field(default_factory=lambda counter=count(): next(counter))
    def __repr__(self):
        return "{ }"

@dataclass(frozen=True)
class SMTIndex:
    value: z3.Int

    def __repr__(self):
        return f"{repr(self.value)}"

@dataclass(frozen=True)
class VarIndex:
    value: int

    def __repr__(self):
        return "ijklmnopqrstuvw"[self.value]

@dataclass(frozen=True)
class LiteralIndex:
    value: int

    def __repr__(self):
        return "xyzt"[self.value]  # if your problem has 5 dimensions you messed up      

class EinSumExpr[T](Protocol):
    is_commutative: bool
    rank: int

    @abstractmethod
    def canon_lt(self, other):
        ...

    @abstractmethod
    def canon_eq(self, other):
        ...

    def struct_eq(self, other):
        return self == other

    def struct_hash(self):
        return hash(self)

    @abstractmethod
    def __repr__(self):
        ...

    @abstractmethod
    def sub_exprs(self) -> Iterable[EinSumExpr[T]]:
    """ Implementation returns list of sub_exprs (whatever this attribute may be called) """
        ...

    @abstractmethod
    def own_indices(self) -> Iterable[T]:
    """ Implementation returns list of own indices """
        ...

    @lru_cache(maxsize=10000)
    def indices(self) -> list[T]: # make sure these are in depth-first/left-to-right order
    """ List all indices """
        return list(self.own_indices()) + [idx for expr in self.sub_exprs() for idx in expr.indices()]
        
    # def map(self, f, ctx):
    #     new_obj = {x: f(v, ctx) for x, v in self.__dict__.items()}
    #     for x, v in self.__dict__.items():
    #         if type(v) != type(new_obj[x]):
    #             raise TypeError(f"{new_obj[x]} is not a plausible replacement for {v} in {str(self.__class__)}.{x}")
    #     return dataclasses.replace(self, **new_obj)
            
    @abstractmethod
    def map[T2](self, f: Callable[[EinSumExpr[T]], EinSumExpr[T2]]) -> EinSumExpr[T2]:
    """ Implementation reconstructs self with new sub_exprs objects (via map) """
        ...

    @abstractmethod
    def map_indices[T2](self, f: Callable[[T], T2]):
    """ Maps indices to new type T2 using f"""
        ...

    def canonical_indexing_problem(self) -> tuple[EinSumExpr[SMTIndex], list[z3.ExprRef]]:
        base_id = f"i{id(self)}"
        next_z3_var = count()
        idx_cache = defaultdict(lambda: z3.Int(f"{base_id}_{next(count)}")
        expr = self.map_indices(lambda idx, ctr=count(): SMTIndex(
                idx_cache[idx] if not isinstance(idx, IndexHole) else
                    z3.Int(f"{base_id}_{next(count)}")
            ))
        
        constraints = []
        def mapper(expr):
            expr, cxs = expr.canonical_indexing_problem()
            constraints += cxs
            return expr
        expr = expr.map(mapper)
        
        if is_commutative:
            duplicates = defaultdict(list)
            for e, e_new in zip(self.sub_exprs(), expr.sub_exprs()):
                duplicates[e].append(e_new)
            for dup_list in duplicates.values():            
                for e, e_next in zip(dup_list, dup_list[1:]):
                    constraints.append(lexico_lt(e.indices(), e_next.indices()))
                    
        return expr, constraints

# @dataclass(frozen=True)
# class EinSumIndex[T]:
#     index: T

    # def canon_lt(self, other):
    #     return self.index < other.index

    # def canon_eq(self, other):
    #     return self.index == other.index

    # def struct_eq(self, other):
    #     return (isinstance(self.index, IndexHole) and 
    #             isinstance(other.index, IndexHole)) or self.index == other.index

    # def struct_hash(self):
    #     return hash(self.value)

    # def __repr__(self):
    #     return f"\{{self.index.repr()}\}"

    # def sub_exprs(self) -> Iterable[EinSumExpr[T]]:
    #     yield from ()

    # def indices(self) -> Iterable[T]:
    #     yield self.index

    # def replace[T2](self, m: dict[T, T2]) -> EinSumExpr[T2]:
    #     return EinSumIndex(m[self.index])

    # def map[T2](self, f: Callable[[T], T2]) -> EinSumExpr[T2]:
    #      return EinSumIndex(f(self.index))

    # def canonical_indexing_problem(self) -> tuple[EinSumExpr[SMTIndex], list[z3.ExprRef]]:
    #     return (EinSumIndex(z3.Int(f'i{id(self)}')), [])

def generate_indexings(expr: EinSumExpr[IndexHole | VarIndex]) -> Iterable[EinSumExpr[VarIndex]]:
    indexed_expr, constraints = expr.canonical_indexing_problem() # includes lexicographic constraints    
    assert_type(indexed_expr, EinSumExpr[SMTIndex])
    # add global constraints
    indices = indexed_expr.indices()
    n_single_inds = expr.rank
    n_total_inds = (len(indices)+n_single_inds)/2
    # use-next-variable constraints
    single_idx_max = 0
    paired_idx_max = n_single_inds
    for j, idx in enumerate(indices):
        s_idx_max_next = z3.Int(f's_idxmax_{j}')
        p_idx_max_next = z3.Int(f'p_idxmax_{j}')
        constraints += [z3.Or(
            z3.And(idx == single_idx_max,
                   s_idx_max_next == idx+1, p_idx_max_next == paired_idx_max)
            z3.And(idx >= n_single_inds, idx <= paired_idx_max,
                   s_idx_max_next == single_idx_max, 
                   p_idx_max_next == paired_idx_max + z3.If(idx==paired_idx_max, 1, 0))
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
    while (result := solver.check()) == z3.sat): # smt solver finds a new solution
        m = solver.model()
        indexing = {index: m[index] for index in indices}
        yield expr.replace(indexing)
        # prevent smt solver from repeating solution
        solver.add(z3.Or(*[idx != val for idx, val in indexing.items()])) 
    if result == z3.unknown:
        raise RuntimeError("Could not solve SMT problem :(")

def lexico_lt(idsA: list[z3.Int], idsB: list[z3.Int]) -> z3.ExprRef:
    lt = False
    for a, b in reversed(zip(idsA, idsB)):
        lt = z3.Or(a < b, z3.And(a == b, lt))
    return lt