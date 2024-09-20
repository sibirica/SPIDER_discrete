# all of the obsolete(?) functions & classes for manipulating unstructured index lists

from dataclasses import dataclass, field, KW_ONLY

### COMMON / CONTINUOUS ###
@dataclass(order=True)
class CompPair(object):
    """
    Data class used to compare integer tuples of space and time derivative orders.

    :attribute torder: time derivative order.
    :attribute xorder: space derivative order.
    :attribute complexity: term complexity.
    """
    torder: int
    xorder: int
    complexity: int = field(init=False)

    def __post_init__(self):
        self.complexity = self.torder + self.xorder

    def __repr__(self):
        return f'({self.torder}, {self.xorder})'

@dataclass
class CompList(object):
    """
    Data class used to compare lists of integer, which don't need to be of equal length.

    :attribute in_list: Index list, a list of integers.
    """
    in_list: List[int]

    def __ge__(self, other):
        """
        Explicitly defines the >= (greater than or equals) operation between two CompList objects.
        Implicitly defines the <= (less than or equals) operation between two CompList objects.
        self is deemed greater than or equals to other if any of the following are true. All elements that share an
        index on self.in_list and other.in_list are the same. The first non-equal pair
        (self.in_list[i], other.in_list[i]) satisfies self.in_list[i] > other.in_list[i].
        NOTE: may be used to calculate every relational operator.

        :param other: CompList object to be compared.
        :return: Comparison result.
        """
        for x, y in zip(self.in_list, other.in_list):
            if x > y:
                return True
            elif x < y:
                return False
        return True

    def special_bigger(self, other):
        """
        A quicker method to compare two CompList objects. Tests if one of the lists contains a zero and the other
        does not.

        :param other: CompList object to be compared.
        :return: Test result.
        """
        if 0 in self.in_list and 0 not in other.in_list:
            return False
        elif 0 in other.in_list and 0 not in self.in_list:
            return True
        else:
            return self >= other

def labels_to_index_list(labels: Dict[int, Union[List[int], Tuple[int]]], n: int) -> List[List[int]]:
    """
    Transform a labels representation of the indexes of a LibraryTerm to its corresponding index_list representation.
    https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels

    :param labels: Dictionary representation of the indexes of a LibraryTerm
    :param n: Number of observables.
    :return: The corresponding index_list.
    """
    index_list = [list() for _ in range(2 * n)]
    for key in sorted(labels.keys()):
        for a in labels[key]:
            index_list[a].append(key)
    return index_list


def index_list_to_labels(index_list: List[List[int]]) -> Dict[int, List[int]]:
    """
    Transforms an index_list representation of the indexes of a LibraryTerm to its corresponding labels representation.
    https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels

    :param index_list: List representation of the indexes of a LibraryTerm
    :return: The corresponding labels dictionary.
    """
    labels = dict()
    for i, li in enumerate(index_list):
        for ind in li:
            if ind in labels.keys():
                labels[ind].append(i)
            else:
                labels[ind] = [i]
    return labels


def flatten(t: List[Union[list, tuple]]) -> list:
    """
    Removes one level of nesting from a List where every item is also a List or Tuple.
    E.g. flatten([[1, 2], [0, 3], [1, 1], (2, 0)]) returns [1, 2, 0, 3, 1, 1, 2, 0].

    :param t: Nested list.
    :return: Flattened list.
    """
    return [item for sublist in t for item in sublist]


def canonicalize_indices(indices: Union[List[int], Tuple[int]]) -> Dict[int, int]:
    """
    Given a flattened list of indices, returns a dictionary that, when applied to every item in that list, canonicalizes
    it.

    :param indices: List of indices.
    :return: Dictionary containing canonicalization rules for the input index list.
    """
    curr_ind = 1
    subs_dict = {0: 0}
    for num in indices:
        if num not in subs_dict.keys():
            subs_dict[num] = curr_ind
            curr_ind += 1
    return subs_dict


def is_canonical(indices: Union[List[int], Tuple[int]]) -> bool:
    """
    Tests if a sequence of indices is in canonical order.

    :param indices: Sequence of indices
    :return: True if indices are canonical, False otherwise.
    """
    subs_dict = canonicalize_indices(indices)
    for key in subs_dict:
        if subs_dict[key] != key:
            return False
    return True


def get_isomorphic_terms(obs_list: list, start_order: Iterable = None) -> Generator[List[int], None, None]:
    """
    Generator that yields all permutations of the obs_list that leave the term unchanged. For example, if a LibraryTerm
    contains two observables that are exactly the same, swapping their order would yield the exact same library term.

    :param obs_list: List of observables (LibraryTerms).
    :param start_order: For internal use, generates new permutations given a starting permutation.
    :return: Generator of all isomorphic permutations.
    """
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


def compress(labels: List[str]) -> List[str]:
    """
    Re-writes an index replacing two repeated terms by the term squared. E.g. ['i','i'] -> ['i^2'].
    For more info on index lists see:
    https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels


    :param labels: List of index strings.
    :return: Compressed list of index strings.
    """
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
def place_pairs(*rank_array,
                min_ind2: int = 0,
                curr_ind: int = 1,
                start: int = 0,
                answer_dict=None) -> Generator[Dict[int, Tuple[int]], None, None]:
    """
    Creates a generator that yields all possible dictionaries of how paired indices are places. For more info on labels
    see:
    https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels

    :param rank_array: A list with 2n elements containing information about n Observable objects. The list stores each
    Observable 's spatial derivative order and tensor rank.
    :param min_ind2: For internal use on recursion.
    :param curr_ind: For internal use on recursion.
    :param start: For internal use on recursion.
    :param answer_dict: For internal use on recursion.
    :return: Generator of labels dictionaries of paired indices.
    """
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


def place_indices(*rank_array) -> Generator[Dict[int, Tuple[int]], None, None]:
    """
    Creates a generator that yields all possible labels dictionaries that would describe 'rank_array'.

    :param rank_array: A list with 2n elements containing information about n Observable objects. The list stores each
    Observable 's spatial derivative order and tensor rank.
    :return: Generator that yields all possible labels, valid or not.
    """
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


# Todo: Add hinting after moving LibraryTensor to commons.
def list_labels(tensor) -> List[Dict[int, Tuple[int]]]:
    """
    Generates a list with all valid label dictionaries for a given LibraryTensor object.
    See test_valid_label() for the definition of a valid label. For more information on labels and indexes see:
    https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels

    :param tensor: A LibraryTensor object.
    :return: List of all valid labels dictionaries for a given LibraryTensor.
    """
    rank_array = []
    for term in tensor.obs_list:
        rank_array.append(term.dorder.xorder)
        if hasattr(term, 'observable'):
            rank_array.append(term.observable.rank)
        else:
            rank_array.append(term.cgp.rank)
    return [output_dict for output_dict in place_indices(*rank_array) if test_valid_label(output_dict, tensor.obs_list)]


# check if index labeling is invalid (i.e. not in non-decreasing order among identical terms)
# this excludes more incorrect options early than is_canonical
# the lexicographic ordering rule fails at N=6 but this is accounted for by the canonicalization
def test_valid_label(output_dict: Dict[int, Union[List[int], Tuple[int]]], obs_list: list) -> bool:
    """
    Checks if a labels dictionary is valid for a given obs_list.
    A label is deemed invalid if its is not in non-decreasing order among identical terms
    For more information regarding labels see:
    https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels

    :param output_dict: A labels dictionary.
    :param obs_list: A list of observables (or any object that supports the == operation).
    :return: False if the label is not in non-decreasing order among identical terms. True otherwise.
    """
    if len(output_dict.keys()) < 2:  # not enough indices for something to be invalid
        return True
    # this can be implemented more efficiently, but the cost is negligible for reasonably small N
    bins: List[list] = []  # bin observations according to equality
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
                if not clist2.special_bigger(clist1):  # if (lexicographic) order decreases OR 'i' appears late
                    return False
    return True


def label_repr(prim: LibraryPrimitive, ind1: List[str], ind2: List[str]) -> str:
    """
    Given a LibraryPrimitive, the list of differential indexes, and its list of observable indexes. Returns a formatted
    string of its representation.
    A Library Primitive is represented by its time derivative(if any) with respective order(if greater than one),
    followed by its spatial derivative (if any) with respective index and order (if greater than one), followed by the
    observable representation with respective index (if any, only for rank 1 tensors).
    NOTE: Indexes are stored in order-sensitive lists. For consistent representation of library terms use canonicalized
    objects.

    :param prim: LibraryPrimitive to be represented
    :param ind1: List of Derivative Indexes
    :param ind2: List of Observable Indexes
    :return: String representation of this indexed LibraryPrimitive.
    """
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













### DISCRETE ###
# as usual: this version of the code will not solve the general problem of observables with rank>2 
class CoarseGrainedPrimitive(GeneralizedObservable):  # represents rho[product of obs_list]
    def __init__(self, obs_list):  # obs_list should be sorted to maintain canonical order
        self.obs_list = obs_list
        self.obs_ranks = [obs.rank for obs in obs_list]  # don't know if we'll need this
        self.rank = sum(self.obs_ranks)
        # add 1 for the coarse-graining operator, rho[1] counts for 1.33
        #self.complexity = (len(obs_list) if obs_list != [] else 0.33) + 1
        self.complexity = len(obs_list) + 1 # rho[1] counts for 1

    def __repr__(self):
        repstr = [str(obs) + ' * ' for obs in self.obs_list]
        return f"rho[{reduce(add, repstr)[:-3]}]" if repstr != [] else "rho"

    def __hash__(self):  # it's nice to be able to use CGP in sets or dicts
        return hash(self.__repr__())

    def index_str(self, obs_dims, coord=False):
        indexed_str = ""
        dim_ind = 0
        for obs, rank in zip(self.obs_list, self.obs_ranks):
            if rank == 0:
                indexed_str += str(obs) + ' * '
            else:
                if coord:  # x/y/z
                    let = dim_to_let[obs_dims[dim_ind]]
                else:  # i/j/k
                    let = num_to_let[obs_dims[dim_ind]]
                indexed_str += f"{str(obs)}_{let} * "
                dim_ind += 1
        # for obs, dims in zip(self.obs_list, obs_dims):
        #    if len(dims) == 0:
        #        indexed_str += str(obs) + ' * '
        #    else:
        #        let = dim_to_let[dim[0]]
        #        indexed_str += f"{str(obs)}_{let} * "
        return f"rho[{indexed_str[:-3]}]" if indexed_str != "" else "rho"

    def __lt__(self, other):
        if not isinstance(other, CoarseGrainedPrimitive):
            raise TypeError("Second argument is not a CoarseGrainedPrimitive.")
        for a, b in zip(self.obs_list, other.obs_list):
            if a == b:
                continue
            else:
                return a < b
        return len(self.obs_list) < len(other.obs_list)

    # TODO: This may be redundant. I believe Python does this process internally.
    def __gt__(self, other):
        if not isinstance(other, CoarseGrainedPrimitive):
            raise TypeError("Second argument is not a CoarseGrainedPrimitive.")
        return other.__lt__(self)

    def __eq__(self, other):
        if not isinstance(other, CoarseGrainedPrimitive):
            raise TypeError("Second argument is not a CoarseGrainedPrimitive.")
        return self.obs_list == other.obs_list

    def __ne__(self, other):
        return not self.__eq__(other)

    def index_canon(self, inds):
        if len(inds) == 0:
            return inds
        new_inds = copy.deepcopy(inds)
        reps = 1
        prev = self.obs_list[0]
        obs_start_ind = 0
        ind_start_ind = 0
        while obs_start_ind < len(self.obs_list) - 1:
            prev = self.obs_list[obs_start_ind]
            while obs_start_ind + reps < len(self.obs_list) and prev == self.obs_list[reps]:
                reps += 1
            if prev.rank > 0:
                new_inds[ind_start_ind:ind_start_ind + reps] = sorted(new_inds[ind_start_ind:ind_start_ind + reps])
            obs_start_ind += reps
            ind_start_ind += reps * prev.rank
        return new_inds

    def is_index_canon(self, inds):  # can just check that inds == index_canon(ind), but this is more efficient
        # print(self.obs_list)
        # print(inds)
        if len(inds) == 0:
            return inds
        reps = 1
        prev = self.obs_list[0]
        obs_start_ind = 0
        ind_start_ind = 0
        while obs_start_ind < len(self.obs_list) - 1:
            prev = self.obs_list[obs_start_ind]
            while obs_start_ind + reps < len(self.obs_list) and prev == self.obs_list[reps]:
                reps += 1
            ni = inds[ind_start_ind:ind_start_ind + reps]
            if prev.rank == 0 or all(a <= b for a, b in zip(ni, ni[1:])):
                obs_start_ind += reps
                ind_start_ind += reps * prev.rank
            else:
                # print('false')
                return False
        # print('true')
        return True
def labels_to_ordered_index_list(labels, ks):
    n = len(ks)
    index_list = [[None] * ks[i] for i in range(n)]
    for key in sorted(labels.keys()):
        for a, b in labels[key]:
            index_list[a][b] = key
    return index_list


def ordered_index_list_to_labels(index_list):
    labels = dict()
    for i, li in enumerate(index_list):
        for j, ind in enumerate(li):
            if ind in labels.keys():
                labels[ind].append((i, j))
            else:
                labels[ind] = [(i, j)]
    return labels

# each label maps to [(bin1, order1), (bin2, order2)], treat sublists of index_list as ordered.
# note: be careful not to modify index_list or labels without remaking because the references are reused
class LibraryTerm(object):
    canon_dict = dict()  # used to store ambiguous canonicalizations (which shouldn't exist for less than 6 indices)

    def __init__(self, libtensor, labels=None, index_list=None):
        self.obs_list = libtensor.obs_list
        self.bin_sizes = flatten([(observable.dorder.xorder, observable.cgp.rank)
                                  for observable in self.obs_list])
        self.libtensor = libtensor
        self.rank = (libtensor.rank % 2)
        self.complexity = libtensor.complexity
        if labels is not None:  # from labels constructor
            self.labels = labels  # dictionary: key = index #, value(s) = location of index among 2n bins
            self.index_list = labels_to_ordered_index_list(labels, self.bin_sizes)
        else:  # from index_list constructor
            self.index_list = index_list
            self.labels = ordered_index_list_to_labels(index_list)
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
                  zip(self.obs_list, self.der_index_list, self.obs_index_list)]
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
            raise TypeError(f"Cannot multiply {type(self)}, {type(other)}")

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
        sorted_ind = flatten(zip(sorted_ind1, sorted_ind2))
        sorted_libtens = LibraryTensor(sorted_obs)
        return LibraryTerm(sorted_libtens, index_list=sorted_ind)

    # noinspection PyUnusedLocal
    def index_canonicalize(self):
        # inc = 0
        # if len(self.labels[0])==2: # if multiple i's, need to increment all indices
        #    inc = 1
        subs_dict = canonicalize_indices(flatten(self.index_list))
        # (a) do index substitutions, (b) sort within sublists 
        new_index_list = [[subs_dict[i] for i in li] for li in self.index_list]
        for li in new_index_list[::2]:  # sort all derivative indices
            li = sorted(li)
        for obs, li in zip(self.obs_list, new_index_list[1::2]):  # canonicalize CGPs
            li = obs.cgp.index_canon(li)
        if all([li1 == li2 for li1, li2 in zip(self.index_list, new_index_list)]):  # no changes were made
            return self
        return LibraryTerm(self.libtensor, index_list=new_index_list)

    def reorder(self, template):
        indexed_zip = zip(self.obs_list, self.der_index_list, self.obs_index_list, template)
        sorted_zip = sorted(indexed_zip, key=lambda x: x[3])
        sorted_obs = [e[0] for e in sorted_zip]
        sorted_ind1 = [e[1] for e in sorted_zip]
        sorted_ind2 = [e[2] for e in sorted_zip]
        sorted_ind = flatten(zip(sorted_ind1, sorted_ind2))
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
        # print(self.obs_list, "->", [term.obs_list for term in ts.term_list])
        # print(self.index_list, "->", [term.index_list for term in ts.term_list])
        return ts.canonicalize()

def label_repr(prim, ind1, ind2):
    torder = prim.dorder.torder
    xorder = prim.dorder.xorder
    cgp = prim.cgp
    if torder == 0:
        tstring = ""
    elif torder == 1:
        tstring = "dt "
    else:
        tstring = f"dt^{torder} "
    if xorder == 0:
        xstring = ""
    else:
        ind1 = [num_to_let[i] for i in ind1]
        ind1 = compress(ind1)
        xlist = [f"d{letter} " for letter in ind1]
        xstring = reduce(add, xlist)
    return tstring + xstring + cgp.index_str(ind2)

def get_valid_reorderings(observables, obs_index_list):
    if len(obs_index_list) == 0:  # don't think this actually happens, but just in case
        yield []
        return
    if len(obs_index_list[0]) == 0:
        for reorder in get_valid_reorderings(observables[1:], obs_index_list[1:]):
            yield [[]] + reorder
            return
    unique_perms = []
    for perm in permutations(obs_index_list[0]):
        if perm not in unique_perms:
            unique_perms.append(perm)
            if observables[0].cgp.is_index_canon(perm):
                for reorder in get_valid_reorderings(observables[1:], obs_index_list[1:]):
                    yield [list(perm)] + reorder

def get_library_terms(tensor, index_list):
    # distribute indices in CGP according to all permutations that are canonical (with only those indices)
    der_index_list = index_list[0::2]
    obs_index_list = index_list[1::2]
    for perm_list in get_valid_reorderings(tensor.obs_list, obs_index_list):
        # print("perm_list:", perm_list)
        # noinspection PyTypeChecker
        yield LibraryTerm(tensor, index_list=flatten(zip(der_index_list, perm_list)))