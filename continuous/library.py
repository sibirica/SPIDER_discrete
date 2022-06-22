import copy
from functools import reduce
from itertools import permutations
from operator import add
from typing import Union, List

import numpy as np
from numpy import inf

from commons.library import *


# noinspection PyArgumentList
@dataclass
class LibraryPrimitive(object):
    """
    Object representing a library primitive. Stores the primitive's derivative order, corresponding Observable, rank and
    complexity.
    """
    dorder: DerivativeOrder
    observable: Observable
    rank: int = field(init=False)
    complexity: int = field(init=False)

    def __post_init__(self):
        self.rank = self.dorder.xorder + self.observable.rank
        self.complexity = self.dorder.complexity + 1

    def __repr__(self):
        tstring, xstring = create_derivative_string(self.dorder.torder, self.dorder.xorder)
        return f'{tstring}{xstring}{self.observable}'

    # For sorting: convention is (1) in ascending order of name/observable, (2) in *ascending* order of dorder

    def __lt__(self, other):
        """
        Explicitly defines the < (lesser than) operator between two LibraryPrimitive objects.
        If two LibraryPrimitive have the same observable, self<other iff self.dorder < other.dorder. Else, compare the
        observables directly.
        :param other: LibraryPrimitive to be compared.
        :return: Test comparison result.
        """
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        if self.observable == other.observable:
            return self.dorder < other.dorder
        else:
            return self.observable < other.observable

    # TODO: This may be redundant. I believe python does this proccess internally.
    def __gt__(self, other):
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        return other.__lt__(self)

    def __eq__(self, other):
        """
        Explicitly defines the == (equals) operation between two LibraryPrimitive objects.
        Two LibraryPrimitive objects are deemed equal if they have the same observable and derivative order.
        :param other:
        :return:
        """
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        return self.observable == other.observable and self.dorder == other.dorder

    def __ne__(self, other):
        return not self.__eq__(other)

    def dt(self) -> 'LibraryPrimitive':
        """
        Increase order of time derivative by one.
        :return: A LibraryPrimitive object with the same spacial order and one plus its temporal order.
        """
        return LibraryPrimitive(self.dorder.dt(), self.observable)

    def dx(self) -> 'LibraryPrimitive':
        """
        Increase order of space derivative by one.
        :return: A LibraryPrimitive object with the same temporal order and one plus its spacial order.
        """
        return LibraryPrimitive(self.dorder.dx(), self.observable)


class IndexedPrimitive(LibraryPrimitive):
    """
    Object representing an IndexedPrimitive. For example the x component of a vector quantity.
    :attribute dorder: DerivativeOrder object representing time and space derivative orders of the primitive used in
    initialization.
    :attribute observable: Observable object represented by this class.
    :attribute rank: int rerpesenting the tensor rank.
    :attribute complexity: number representing the complexity score of the object.
    :attribute ndims: number of spatial-temporal dimensions.
    :attribute nderivs: sum of all derivative orders.
    :attribute obs_dim: Integer representing the Dimension of the observable/tensor. For example the x component of a
    velocity field would have this value set to 0.
    """
    dim_to_let = {0: 'x', 1: 'y', 2: 'z'}  # Dimensions to letter dictionary

    def __init__(self, prim: Union[LibraryPrimitive, 'IndexedPrimitive'],
                 space_orders: List[int] = None,
                 obs_dim: int = None,
                 newords: List[int] = None):
        """

        :param prim: Primitive to which initialize the class. It may be a LibraryPrimitive or an IndexedPrimitive.
        :param space_orders: List containing the order of the spatial derivatives. Ex: [1,2,3] would represent a first
        order derivative in x, a second order derivative in y, and a third order derivative in z. Only applied when
        initializng from a LibraryPrimitve.
        :param obs_dim: Integer representing the Dimension of the observable/tensor. For example the x component of a
        velocity field would have this value set to 0. Only applied when initializng from a LibraryPrimitve.
        :param newords: List containing the order of the spatial and time derivatives. Ex: [1,2,3,0] would represent a
        first order derivative in x, a second order derivative in y, a third order derivative in z, and no time
        derivatives. Only used when initialized from another IndexedPrimitive.
        """
        self.dorder = prim.dorder  # DerivativeOrder object representing time and space derivative orders of prim.
        self.observable = prim.observable
        self.rank = prim.rank
        self.complexity = prim.complexity
        if newords is None:  # normal constructor
            self.dimorders = space_orders + [self.dorder.torder]
            self.obs_dim = obs_dim
        else:  # modifying constructor
            self.dimorders = newords
            self.obs_dim = prim.obs_dim
        self.ndims = len(self.dimorders)
        self.nderivs = sum(self.dimorders)

    def __repr__(self):
        """
        IndexedPrimitives are represented as 'dx^idy^jdz^kdt^lO' where O stands for the observable. If the derivative
        orders (a.k.a. i,j,k,l) are 1, they are ommited, if they are 0 the whole derivative term is ommited.
        """
        torder = self.dimorders[-1]
        xstring = ""
        for i in range(len(self.dimorders) - 1):
            let = self.dim_to_let[i]
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
        if self.obs_dim is None:
            dimstring = ""
        else:
            let = self.dim_to_let[self.obs_dim]
            dimstring = f"_{let}"
        return f'{tstring}{xstring}{self.observable}{dimstring}'

    def __eq__(self, other):
        """
        Two IndexedPrimitive are deemed equal if they have the same dimension order, observable being represented, and
        indexed dimension (a.k.a. they correspond to the x component).
        """
        return (self.dimorders == other.dimorders and self.observable == other.observable
                and self.obs_dim == other.obs_dim)

    def __mul__(self, other):
        if isinstance(other, IndexedTerm):
            return IndexedTerm(obs_list=[self] + other.obs_list)
        else:
            return IndexedTerm(obs_list=[self, other])

    def succeeds(self, other, dim):
        """
        Tests if other is a derivative of self in the given dimension (dim).
        """
        copyorders = self.dimorders.copy()
        copyorders[dim] += 1
        return copyorders == other.dimorders and self.observable == other.observable and self.obs_dim == other.obs_dim

    def diff(self, dim):
        """
        Returns an IndexedPrimitive with the same properties as self but with an extra derivative order in the given
        dimension (dim).
        :param dim: Dimension to take the derivative.
        :return: IndexedPrimitive with a higher derivative order.
        """
        newords = self.dimorders.copy()
        newords[dim] += 1
        return IndexedPrimitive(self, newords=newords)


class LibraryTensor(object):  # unindexed version of LibraryTerm
    def __init__(self, observables):
        # constructor for library terms consisting of a primitive with some derivatives
        if isinstance(observables,
                      LibraryPrimitive):
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
            raise ValueError(f"Cannot multiply {type(self)}, {type(other)}")

    def __rmul__(self, other):
        return self*other

    def __repr__(self):
        repstr = [str(obs) + ' * ' for obs in self.obs_list]
        return reduce(add, repstr)[:-3]


def flatten(t):
    return [item for sublist in t for item in sublist]


num_to_let_dict = {0: 'i', 1: 'j', 2: 'k', 3: 'l', 4: 'm', 5: 'n', 6: 'p'}
let_to_num_dict = {v: k for k, v in num_to_let_dict.items()}  # inverted dict


def num_to_let(num_list):
    return [[num_to_let_dict[i] for i in li] for li in num_list]


def canonicalize_indices(indices):
    curr_ind = 1
    subs_dict = {0: 0}
    for num in indices:
        if num not in subs_dict.keys():
            subs_dict[num] = curr_ind
            curr_ind += 1
    return subs_dict


def is_canonical(indices):
    subs_dict = canonicalize_indices(indices)
    for key in subs_dict:
        if subs_dict[key] != key:
            return False
    return True


# note: be careful not to modify index_list or labels without remaking because the references are reused
class LibraryTerm(object):
    canon_dict = dict()  # used to store ambiguous canonicalizations (which shouldn't exist for less than 6 indices)

    def __init__(self, libtensor, labels=None, index_list=None):
        self.obs_list = libtensor.obs_list
        self.libtensor = libtensor
        self.rank = (libtensor.rank % 2)
        self.complexity = libtensor.complexity
        if labels is not None:  # from labels constructor
            self.labels = labels  # dictionary: key = index #, value(s) = location of index among 2n bins
            self.index_list = labels_to_index_list(labels, len(self.obs_list))
        else:  # from index_list constructor
            self.index_list = index_list
            self.labels = index_list_to_labels(index_list)
        self.der_index_list = self.index_list[0::2]
        self.obs_index_list = self.index_list[1::2]
        self.is_canonical = None

    def __add__(self, other):
        if isinstance(other, LibraryTerm):
            return TermSum([self, other])
        else:
            return TermSum([self] + other.term_list)

    def __eq__(self, other):
        if isinstance(other, LibraryTerm):
            return self.obs_list == other.obs_list and self.index_list == other.index_list
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __repr__(self):
        repstr = [label_repr(obs, ind1, ind2) + ' * ' for (obs, ind1, ind2) in
                  zip(self.obs_list, num_to_let(self.der_index_list), num_to_let(self.obs_index_list))]
        return reduce(add, repstr)[:-3]

    def __hash__(self):  # it's nice to be able to use LibraryTerms in sets or dicts
        return hash(self.__repr__())

    def __mul__(self, other):
        if isinstance(other, LibraryTerm):
            if self.rank < other.rank:
                return other.__mul__(self)
            if len(self.labels.keys()) > 0:
                shift = max(self.labels.keys())
            else:
                shift = 0
            if other.rank == 1:
                a, b = self.increment_indices(1), other.increment_indices(shift + 1)
            else:
                a, b = self, other.increment_indices(shift)
            return LibraryTerm(LibraryTensor(a.obs_list + b.obs_list),
                               index_list=a.index_list + b.index_list).canonicalize()
        elif str(other) == "1":
            return self
        elif isinstance(other, Equation):
            return other.__mul__(self)
        else:
            raise ValueError(f"Cannot multiply {type(self)}, {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def structure_canonicalize(self):
        indexed_zip = zip(self.obs_list, self.der_index_list, self.obs_index_list)
        sorted_zip = sorted(indexed_zip, key=lambda x: x[0])
        if sorted_zip == indexed_zip:  # no changes necessary
            return self
        sorted_obs = [e[0] for e in sorted_zip]
        sorted_ind1 = [e[1] for e in sorted_zip]
        sorted_ind2 = [e[2] for e in sorted_zip]
        sorted_ind = flatten(list(zip(sorted_ind1, sorted_ind2)))
        sorted_libtens = LibraryTensor(sorted_obs)
        return LibraryTerm(sorted_libtens, index_list=sorted_ind)

    def index_canonicalize(self):
        # inc = 0
        # if len(self.labels[0])==2: # if multiple i's, need to increment all indices
        #    inc = 1
        subs_dict = canonicalize_indices(flatten(self.index_list))
        new_index_list = [[subs_dict[i] for i in li] for li in self.index_list]
        if all([li1 == li2 for li1, li2 in zip(self.index_list, new_index_list)]):  # no changes were made
            return self
        return LibraryTerm(self.libtensor, index_list=new_index_list)

    def reorder(self, template):
        indexed_zip = zip(self.obs_list, self.der_index_list, self.obs_index_list, template)
        sorted_zip = sorted(indexed_zip, key=lambda x: x[3])
        sorted_obs = [e[0] for e in sorted_zip]
        sorted_ind1 = [e[1] for e in sorted_zip]
        sorted_ind2 = [e[2] for e in sorted_zip]
        sorted_ind = flatten(list(zip(sorted_ind1, sorted_ind2)))
        sorted_libtens = LibraryTensor(sorted_obs)
        return LibraryTerm(sorted_libtens, index_list=sorted_ind)

    def canonicalize(self):  # return canonical representation and set is_canonical flag (used to determine if valid)
        str_canon = self.structure_canonicalize()
        if str_canon in self.canon_dict:
            canon = self.canon_dict[str_canon]
            self.is_canonical = (self == canon)
            return canon
        reorderings = []
        alternative_canons = []
        for template in get_isomorphic_terms(str_canon.obs_list):
            term = str_canon.reorder(template)
            if term not in reorderings:  # exclude permutation-symmetric options
                reorderings.append(term)
                canon_term = term.index_canonicalize()
                alternative_canons.append(canon_term)
        canon = min(alternative_canons, key=str)
        for alt_canon in alternative_canons:
            self.canon_dict[alt_canon] = canon
            self.is_canonical = (self == canon)
        return canon

    def increment_indices(self, inc):
        index_list = [[index + inc for index in li] for li in self.index_list]
        return LibraryTerm(LibraryTensor(self.obs_list), index_list=index_list)

    def dt(self):
        terms = []
        for i, obs in enumerate(self.obs_list):
            new_obs = obs.dt()
            # note: no need to recanonicalize terms after a dt
            lt = LibraryTerm(LibraryTensor(self.obs_list[:i] + [new_obs] + self.obs_list[i + 1:]),
                             index_list=self.index_list)
            terms.append(lt)
        ts = TermSum(terms)
        return ts.canonicalize()

    def dx(self):
        terms = []
        for i, obs in enumerate(self.obs_list):
            new_obs = obs.dx()
            new_index_list = copy.deepcopy(self.index_list)
            new_index_list[2 * i].insert(0, 0)
            lt = LibraryTerm(LibraryTensor(self.obs_list[:i] + [new_obs] + self.obs_list[i + 1:]),
                             index_list=new_index_list)
            if lt.rank == 0:
                lt = lt.increment_indices(1)
            lt = lt.canonicalize()  # structure changes after derivative so we must recanonicalize
            terms.append(lt)
        ts = TermSum(terms)
        return ts.canonicalize()


def get_isomorphic_terms(obs_list, start_order=None):
    if start_order is None:
        start_order = list(range(len(obs_list)))
    if len(obs_list) == 0:
        yield []
        return
    reps = 1
    prev = obs_list[0]
    while reps < len(obs_list) and prev == obs_list[reps]:
        reps += 1
    for new_list in get_isomorphic_terms(obs_list[reps:], start_order[reps:]):
        for perm in permutations(start_order[:reps]):
            yield list(perm) + new_list


class IndexedTerm(object):  # LibraryTerm with i's mapped to x/y/z
    def __init__(self, libterm=None, space_orders=None, obs_dims=None, obs_list=None):
        if obs_list is None:  # normal "from scratch" constructor
            self.rank = libterm.rank
            self.complexity = libterm.complexity
            # self.obs_dims = obs_dims
            nterms = len(libterm.obs_list)
            self.obs_list = libterm.obs_list.copy()
            for i, obs, sp_ord, obs_dim in zip(range(nterms), libterm.obs_list, space_orders, obs_dims):
                self.obs_list[i] = IndexedPrimitive(obs, sp_ord, obs_dim)
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

    def drop(self, obs):
        # print(self.obs_list)
        obs_list_copy = self.obs_list.copy()
        if len(obs_list_copy) > 1:
            obs_list_copy.remove(obs)
        else:
            obs_list_copy = []
        return IndexedTerm(obs_list=obs_list_copy)

    def diff(self, dim):
        for i, obs in enumerate(self.obs_list):
            yield obs.diff(dim) * self.drop(obs)


# Note: must be handled separately in derivatives
class ConstantTerm(IndexedTerm):
    def __init__(self):
        self.obs_list = []
        self.rank = 0
        self.complexity = 1

    def __repr__(self):
        return "1"


def label_repr(prim, ind1, ind2):
    torder = prim.dorder.torder
    xorder = prim.dorder.xorder
    obs = prim.observable
    if torder == 0:
        tstring = ""
    elif torder == 1:
        tstring = "dt "
    else:
        tstring = f"dt^{torder} "
    if xorder == 0:
        xstring = ""
    else:
        ind1 = compress(ind1)
        xlist = [f"d{letter} " for letter in ind1]
        xstring = reduce(add, xlist)
    if obs.rank == 1:
        obstring = obs.string + "_" + ind2[0]
    else:
        obstring = obs.string
    return tstring + xstring + obstring


def compress(labels):
    local_copy = []
    skip = False
    for i in range(len(labels)):
        if i < len(labels) - 1 and labels[i] == labels[i + 1]:
            local_copy.append(labels[i] + '^2')
            skip = True
        elif not skip:
            local_copy.append(labels[i])
        else:
            skip = False
    return local_copy


# make a dictionary of how paired indices are placed
def place_pairs(*rank_array, min_ind2=0, curr_ind=1, start=0, answer_dict=None):
    if answer_dict is None:
        answer_dict = dict()
    while rank_array[start] <= 0:
        start += 1
        min_ind2 = 0
        if start >= len(rank_array):
            yield answer_dict
            return
    ind1 = start
    for ind2 in range(min_ind2, len(rank_array)):
        if (ind1 == ind2 and rank_array[ind1] == 1) or rank_array[ind2] == 0:
            continue
        min_ind2 = ind2
        dict1 = answer_dict.copy()
        dict1.update({curr_ind: (ind1, ind2)})
        copy_array = np.array(rank_array)
        copy_array[ind1] -= 1
        copy_array[ind2] -= 1
        yield from place_pairs(*copy_array, min_ind2=min_ind2, curr_ind=curr_ind + 1, start=start, answer_dict=dict1)


def place_indices(*rank_array):
    # only paired indices allowed
    if sum(rank_array) % 2 == 0:
        yield from place_pairs(*rank_array)
    # one single index
    else:
        for single_ind in range(len(rank_array)):
            if rank_array[single_ind] > 0:
                copy_array = np.array(rank_array)
                copy_array[single_ind] -= 1
                yield from place_pairs(*copy_array, answer_dict={0: (single_ind,)})


def list_labels(tensor):
    rank_array = []
    for term in tensor.obs_list:
        rank_array.append(term.dorder.xorder)
        rank_array.append(term.observable.rank)
    return [output_dict for output_dict in place_indices(*rank_array) if test_valid_label(output_dict, tensor.obs_list)]


# check if index labeling is invalid (i.e. not in non-decreasing order among identical terms)
# this excludes more incorrect options early than is_canonical
# the lexicographic ordering rule fails at N=6 but this is accounted for by the canonicalization
def test_valid_label(output_dict, obs_list):
    if len(output_dict.keys()) < 2:  # not enough indices for something to be invalid
        return True
    # this can be implemented more efficiently, but the cost is negligible for reasonably small N
    bins = []  # bin observations according to equality
    for obs in obs_list:
        found_match = False
        for bi in bins:
            if bi is not None and obs == bi[0]:
                bi.append(obs)
                found_match = True
        if not found_match:
            bins.append([obs])
    if len(bins) == len(obs_list):
        return True  # no repeated values
    # else need to check more carefully
    n = len(obs_list)
    index_list = labels_to_index_list(output_dict, n)
    for i in range(n):
        for j in range(i + 1, n):
            if obs_list[i] == obs_list[j]:
                clist1 = CompList(index_list[2 * i] + index_list[2 * i + 1])
                clist2 = CompList(index_list[2 * j] + index_list[2 * j + 1])
                if not clist2.special_bigger(clist1):  # if (lexicographic) order decreases OR i appears late
                    return False
    return True


def raw_library_tensors(observables, obs_orders, nt, nx, max_order=DerivativeOrder(inf, inf), zeroidx=0):
    # print(obs_orders, nt, nx, max_order)
    while obs_orders[zeroidx] == 0:
        zeroidx += 1
        if zeroidx == len(observables):
            yield 1
            return
    if sum(obs_orders) == 1:
        i = obs_orders.index(1)
        do = DerivativeOrder(nt, nx)
        if max_order >= do:
            prim = LibraryPrimitive(do, observables[i])
            yield LibraryTensor(prim)
        return
    for i in range(nt + 1):
        for j in range(nx + 1):
            do = DerivativeOrder(i, j)
            if max_order >= do:
                prim = LibraryPrimitive(do, observables[zeroidx])
                term1 = LibraryTensor(prim)
                new_orders = list(obs_orders)
                new_orders[zeroidx] -= 1
                if obs_orders[zeroidx] == 1:  # reset max_order since we are going to next terms
                    do = DerivativeOrder(inf, inf)
                for term2 in raw_library_tensors(observables, new_orders, nt - i, nx - j, do, zeroidx):
                    yield term2 * term1  # reverse back to ascending order here


rho = Observable('rho', 0)
v = Observable('v', 1)


def generate_terms_to(order, observables=None, max_observables=999):
    if observables is None:
        observables = [rho, v]
    observables = sorted(observables, reverse=True)  # make sure ordering is reverse of canonicalization rules
    libterms = list()
    libterms.append(ConstantTerm())
    n = order  # max number of "blocks" to include
    k = len(observables)
    # not a valid term if no observables or max exceeded
    for part in partition(n, k + 2):  # k observables + 2 derivative dimensions
        # print(part)
        if 0 < sum(part[:k]) <= max_observables:
            nt, nx = part[-2:]
            obs_orders = part[:-2]
            for tensor in raw_library_tensors(observables, obs_orders, nt, nx):
                for label in list_labels(tensor):
                    lt = LibraryTerm(tensor, label)
                    canon = lt.canonicalize()
                    if lt.is_canonical:
                        libterms.append(lt)
    return libterms


def partition(n, k):
    """n is the integer to partition up to, k is the length of partitions"""
    if k < 1:
        return
    if k == 1:
        for i in range(n + 1):
            yield i,
        return
    for i in range(n + 1):
        for result in partition(n - i, k - 1):
            yield (i,) + result


class Equation(object):  # can represent equation (expression = 0) OR expression
    def __init__(self, term_list, coeffs):  # terms are LibraryTerms, coeffs are real numbers
        content = zip(term_list, coeffs)
        sorted_content = sorted(content, key=lambda x: x[0])
        # note that sorting guarantees canonicalization in equation term order
        self.term_list = [e[0] for e in sorted_content]
        self.coeffs = [e[1] for e in sorted_content]
        self.rank = term_list[0].rank
        self.complexity = sum([term.complexity for term in term_list])  # another choice is simply the number of terms

    def __add__(self, other):
        if isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise ValueError(f"Second argument {other}) is not an equation.")

    def __rmul__(self, other):
        if isinstance(other, LibraryTerm):
            return Equation([other * term for term in self.term_list], self.coeffs)
        else:  # multiplication by number
            return Equation(self.term_list, [other * c for c in self.coeffs])

    def __mul__(self, other):
        return self.__rmul__(other)

    def __repr__(self):
        repstr = [str(coeff) + ' * ' + str(term) + ' + ' for term, coeff in zip(self.term_list, self.coeffs)]
        return reduce(add, repstr)[:-3]

    def __str__(self):
        return self.__repr__() + " = 0"

    def __eq__(self, other):
        for term, ot in zip(self.term_list, other.term_list):
            if term != ot:
                return False
        for coeff, ot in zip(self.coeffs, other.coeffs):
            if term != ot:
                return False
        return True

    def dt(self):
        components = [coeff * term.dt() for term, coeff in zip(self.term_list, self.coeffs)
                      if not isinstance(term, ConstantTerm)]
        return reduce(add, components).canonicalize()

    def dx(self):
        components = [coeff * term.dx() for term, coeff in zip(self.term_list, self.coeffs)
                      if not isinstance(term, ConstantTerm)]
        return reduce(add, components).canonicalize()

    def canonicalize(self):
        if len(self.term_list) == 0:
            return self
        term_list = []
        coeffs = []
        i = 0
        while i < len(self.term_list):
            reps = 0
            prev = self.term_list[i]
            while i < len(self.term_list) and prev == self.term_list[i]:
                reps += self.coeffs[i]
                i += 1
            term_list.append(prev)
            coeffs.append(reps)
        return Equation(term_list, coeffs)

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
        if isinstance(other, TermSum):
            return TermSum(self.term_list + other.term_list, self.coeffs + other.coeffs)
        elif isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise ValueError(f"Second argument {other}) is not an equation.")
