import copy
from functools import reduce
from operator import add
from itertools import permutations
from numpy import inf
import numpy as np

from commons.library import *



# noinspection PyArgumentList
@dataclass
class LibraryPrimitive(object):
    dorder: DerivativeOrder
    cgp: CoarseGrainedPrimitive
    rank: int = field(init=False)
    complexity: int = field(init=False)

    def __post_init__(self):
        self.rank = self.dorder.xorder + self.cgp.rank
        self.complexity = self.dorder.complexity + self.cgp.complexity

    def __repr__(self):
        tstring, xstring = create_derivative_string(self.dorder.torder, self.dorder.xorder)
        return f'{tstring}{xstring}{self.cgp}'

    def __hash__(self):
        return hash(self.__repr__())

    # For sorting: convention is (1) in ascending order of name, (2) in *ascending* order of dorder

    def __lt__(self, other):
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        if self.cgp == other.cgp:
            return self.dorder < other.dorder
        else:
            return self.cgp < other.cgp

    def __gt__(self, other):
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        return other.__lt__(self)

    def __eq__(self, other):
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        return self.cgp == other.cgp and self.dorder == other.dorder

    def __ne__(self, other):
        return not self.__eq__(other)

    def dt(self):
        return LibraryPrimitive(self.dorder.dt(), self.cgp)

    def dx(self):
        return LibraryPrimitive(self.dorder.dx(), self.cgp)

    def index_canon(self, inds): # FOR POLYMORPHIC COMPATIBILITY
        return self.cgp.index_canon(inds)

# (1) Evaluation will need some rework to account for repetitions both within derivatives and coarse-grained primitive
class IndexedPrimitive(LibraryPrimitive):
    def __init__(self, prim, space_orders=None, obs_dims=None, newords=None):
        # obs_dims should be a flat list
        # however, it will be converted tо nested list where inner lists correspond to indices of observable
        self.dorder = prim.dorder
        self.cgp = prim.cgp
        self.rank = prim.rank
        self.complexity = prim.complexity
        if newords is None:  # normal constructor
            self.dimorders = space_orders + [self.dorder.torder]
            self.obs_dims = obs_dims
        else:  # modifying constructor
            self.dimorders = newords
            self.obs_dims = prim.obs_dims
        self.ndims = len(self.dimorders)
        self.nderivs = sum(self.dimorders)

    def __repr__(self):
        torder = self.dimorders[-1]
        xstring = ""
        for i in range(len(self.dimorders) - 1):
            let = dim_to_let[i]
            xorder = self.dimorders[i]
            if xorder == 0:
                xstring += ""
            elif xorder == 1:
                xstring += f"d{let} "
            else:
                xstring += f"d{let}^{xorder} "
        if torder == 0:
            tstring = ""
        elif torder == 1:
            tstring = "dt "
        else:
            tstring = f"dt^{torder} "
        return f'{tstring}{xstring}{self.cgp.index_str(self.obs_dims, coord=True)}'

    def __hash__(self):
        #return hash(self.__repr__())
        return hash(self.cgp.index_str(self.obs_dims, coord=True))
        
    def __eq__(self, other):
        return (self.dimorders == other.dimorders and self.cgp == other.cgp
                and self.obs_dims == other.obs_dims)

    def succeeds(self, other, dim):
        copyorders = copy.deepcopy(self.dimorders)
        copyorders[dim] += 1
        return copyorders == other.dimorders and self.cgp == other.cgp and self.obs_dims == other.obs_dims

    def diff(self, dim):
        newords = copy.deepcopy(self.dimorders)
        newords[dim] += 1
        return IndexedPrimitive(self, newords=newords)

    def __mul__(self, other):
        if isinstance(other, IndexedTerm):
            return IndexedTerm(obs_list=[self] + other.obs_list)
        else:
            return IndexedTerm(obs_list=[self] + [other])


class LibraryTensor(object):  # unindexed version of LibraryTerm
    def __init__(self, observables):
        if isinstance(observables,
                      LibraryPrimitive):  # constructor for library terms consisting of a primitive w/ some derivatives
            self.obs_list = [observables]
        else:  # constructor for library terms consisting of a product
            self.obs_list = observables
        self.rank = sum([obs.rank for obs in self.obs_list])
        self.complexity = sum([obs.complexity for obs in self.obs_list])

    def __mul__(self, other):
        if isinstance(other, LibraryTensor):
            return LibraryTensor(self.obs_list + other.obs_list)
        elif other == 1:
            return self
        else:
            raise TypeError(f"Cannot multiply {type(self)}, {type(other)}")

    def __rmul__(self, other):
        if other != 1:
            return other.__mul__(self)
        else:
            return self

    def __repr__(self):
        repstr = [str(obs) + ' * ' for obs in self.obs_list]
        return reduce(add, repstr)[:-3]

class IndexedTerm(object):  # LibraryTerm with i's mapped to x/y/z
    def __init__(self, libterm=None, space_orders=None, nested_obs_dims=None, obs_list=None):
        if obs_list is None:  # normal "from scratch" constructor
            self.rank = libterm.rank
            self.complexity = libterm.complexity
            nterms = len(libterm.obs_list)
            self.obs_list = copy.deepcopy(libterm.obs_list)
            for i, obs, sp_ord, obs_dims in zip(range(nterms), libterm.obs_list, space_orders, nested_obs_dims):
                self.obs_list[i] = IndexedPrimitive(obs, sp_ord, obs_dims)
            self.ndims = len(space_orders[0]) + 1
            self.nderivs = np.max([p.nderivs for p in self.obs_list])
        else:  # direct constructor from observable list
            # print(obs_list)
            if len(obs_list) > 0:  # if term is not simply equal to 1
                self.rank = obs_list[0].rank
                self.ndims = obs_list[0].ndims
                self.obs_list = obs_list
                self.complexity = sum([obs.complexity for obs in obs_list])
                self.nderivs = np.max([p.nderivs for p in self.obs_list])
            else:
                self.obs_list = []
                self.ndims = 0
                self.nderivs = 0
                self.complexity = 0

    def __repr__(self):
        repstr = [str(obs) + ' * ' for obs in self.obs_list]
        return reduce(add, repstr)[:-3]

    def __mul__(self, other):
        if isinstance(other, IndexedTerm):
            return IndexedTerm(obs_list=self.obs_list + other.obs_list)
        else:
            return IndexedTerm(obs_list=self.obs_list + [other])

    def drop(self, obs):  # remove one instance of obs
        obs_list_copy = copy.deepcopy(self.obs_list)
        if len(obs_list_copy) > 1:
            obs_list_copy.remove(obs)
        else:
            obs_list_copy = []
        return IndexedTerm(obs_list=obs_list_copy)

    def drop_all(self, obs):  # remove *aLL* instances of obs
        if len(self.obs_list) > 1:
            obs_list_copy = list(filter(obs.__ne__, self.obs_list))
        else:
            obs_list_copy = []
        return IndexedTerm(obs_list=obs_list_copy)

    def diff(self, dim):
        for i, obs in enumerate(self.obs_list):
            yield obs.diff(dim) * self.drop(obs)

# return LibraryTensors with fixed allocation of complexity
def raw_library_tensors(observables, orders, max_order=None, zeroidx=0): 
    # basically: iteratively take any possible subset from [obs_orders; nrho; nt; nx] 
    # as long as it's lexicographically less than previous order; take at least one of first observable

    # print(orders, max_order, zeroidx)
    n = len(observables)
    if orders[n] == 0:
        if sum(orders) > 0:  # invalid distribution
            return
        else:
            yield 1
            return
    while zeroidx < n and orders[zeroidx] == 0:
        zeroidx += 1
    if zeroidx < n:
        orders[zeroidx] -= 1  # always put in at least one of these to maintain lexicographic order

    # orders = obs_orders + [nt, nx]
    # print("ORDERS: ", orders)
    for tup in yield_legal_tuples(orders[:n] + [0] + orders[n + 1:]):  # ignore the rho index which is deducted automatically
        orders_copy = orders.copy()
        popped_orders = list(tup)
        # print("Popped: ", popped_orders)
        for i in range(len(orders)):
            orders_copy[i] -= popped_orders[i]
        if sum(orders_copy[:-2]) == 0 and sum(
                orders_copy[-2:]) > 0:  # all observables + rho popped but derivatives remain
            continue  # otherwise we will have duplicates from omitting derivatives
        if zeroidx < n:
            popped_orders[zeroidx] += 1  # re-adding the one
        orders_copy[n] -= 1  # account for the rho we used
        popped_orders[n] += 1  # include the rho here as well
        po_cl = CompList(popped_orders)
        if max_order is None or po_cl <= max_order:
            obs_list = []
            for i, order in enumerate(popped_orders[:-3]):  # rho appears automatically so stop at -3
                obs_list += [observables[i]] * order
            cgp = CoarseGrainedPrimitive(obs_list[::-1])  # flip order of observables back to ascending
            do = DerivativeOrder(popped_orders[-2], popped_orders[-1])
            prim = LibraryPrimitive(do, cgp)
            term1 = LibraryTensor(prim)
            # for term2 in raw_library_tensors(observables, orders[:-2], orders[-2], orders[-1], max_order=max_order):
            for term2 in raw_library_tensors(observables, orders_copy, po_cl, zeroidx):
                yield term2 * term1  # reverse order here to match canonicalization rules!

def generate_terms_to(order, observables=None, max_observables=999, max_rho=999):
    # note: this ignores the fact that rho operator adds complexity, but you can filter by complexity later
    observables = sorted(observables, reverse=True)  # ordering opposite of canonicalization rules for now
    libterms = list()
    libterms.append(ConstantTerm())
    n = order  # max number of "blocks" to include
    k = len(observables)
    partitions = partition(n, k + 3)  # k observables + rho + 2 derivative dimensions
    # not a valid term if no observables or max exceeded
    for part in partitions:
        if part[k] > 0 and sum(part[:k]) <= max_observables and part[k] <= max_rho:  # popped a rho, did not exceed max observables
            # nt, nx = part[-2:]
            # obs_orders = part[:-2]
            # for tensor in raw_library_tensors(observables, obs_orders, nt, nx):
            # print("\n\n\n")
            # print("Partition:", part)
            for tensor in raw_library_tensors(observables, list(part)):
                if tensor.complexity <= order:  # this may not be true since we set complexity of rho[1]>1
                    # print("Tensor", tensor)
                    # print("List of labels", list_labels(tensor))
                    for label in list_labels(tensor):
                        # print("Label", label)
                        index_list = labels_to_index_list(label, len(tensor.obs_list))
                        # print("Index list", index_list)
                        for lt in get_library_terms(tensor, index_list):
                            # print("LT", lt)
                            # note: not sure where to put this check
                            canon = lt.canonicalize()
                            if lt.is_canonical:
                                # print("is canonical")
                                libterms.append(lt)
    return libterms