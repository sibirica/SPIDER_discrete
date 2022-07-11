from dataclasses import dataclass, field
from typing import List, Dict, Union
from itertools import permutations

import numpy as np


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
    :attribute in_list: Internal (or index?) list, a list of integers.
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


class DerivativeOrder(CompPair):
    """
    Object to store and manipulate derivative orders.
    """

    def dt(self) -> 'DerivativeOrder':
        """
        Increase order of time derivative by one.
        :return: A Derivative Order object with the same spacial order and one plus its temporal order.
        """
        return DerivativeOrder(self.torder + 1, self.xorder)

    def dx(self) -> 'DerivativeOrder':
        """
        Increase order of space derivative by one.
        :return: A Derivative Order object with the same temporal order and one plus its spacial order.
        """
        return DerivativeOrder(self.torder, self.xorder + 1)


@dataclass
class Observable(object):
    """
    Data class object that stores a string representation of an observable as well as its rank. For documentation
    purposes, this class will always be refered to as 'Observable' (capitalized), unless stated otherwise. Furthermore,
    the term 'observable' usually does NOT refer to this class, but rather to a LibraryPrimitive object.
    :attribute string: String representation of the observable.
    :attribute rank: Tensor rank of the observable.
    """
    string: str  # String representing the Observable.
    rank: int  # Derivative rank of the Observable.

    def __repr__(self):
        return self.string

    # For sorting: convention is in ascending order of name

    def __lt__(self, other):
        if not isinstance(other, Observable):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return self.string < other.string

    # TODO: This may be redundant. I believe python does this proccess internally.
    def __gt__(self, other):
        if not isinstance(other, Observable):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return other.__lt__(self)

    def __eq__(self, other):
        if not isinstance(other, Observable):
            raise TypeError(f"Operation not supported between instances of '{type(self)}' and '{type(other)}'")
        return self.string == other.string

    def __ne__(self, other):
        return not self.__eq__(other)


def create_derivative_string(torder: int, xorder: int) -> (str, str):
    """
    Creates a derivative string given a temporal order and a spatial order.
    :param torder: Temporal derivative order.
    :param xorder: Spatial Derivative Order.
    :return: Time derivative string, Spatial derivative string
    """

    if torder == 0:
        tstring = ""
    elif torder == 1:
        tstring = "dt "
    else:
        tstring = f"dt^{torder} "
    if xorder == 0:
        xstring = ""
    elif xorder == 1:
        xstring = "dx "
    else:
        xstring = f"dx^{xorder} "

    return tstring, xstring


def labels_to_index_list(labels: Dict[int, List[int]], n: int) -> List[List[int]]:
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


def flatten(t: List[Union[list, tuple]]):
    """
    Removes one level of nesting from a List where every item is also a List or Tuple.
    E.g. flatten([[1, 2], [0, 3], [1, 1], (2, 0)]) returns [1, 2, 0, 3, 1, 1, 2, 0].
    :param t: Nested list.
    :return: Flattened list.
    """
    return [item for sublist in t for item in sublist]


num_to_let_dict = {0: 'i', 1: 'j', 2: 'k', 3: 'l', 4: 'm', 5: 'n', 6: 'p'}
let_to_num_dict = {v: k for k, v in num_to_let_dict.items()}  # inverted dict


def num_to_let(num_list: List[List[int]]) -> List[List[str]]:
    """
    Transforms a list of lists of int indexes into the respective list of lists of letter indexes.
    :param num_list: List of lists of int indexes.
    :return: Corresponding list of lists of str indexes.
    """
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


rho = Observable('rho', 0)
v = Observable('v', 1)

