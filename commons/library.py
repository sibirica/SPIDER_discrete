from dataclasses import dataclass, field
from typing import List


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
    Data class object that stores a string representation of an observable as well as its rank.
    """
    string: str  # String representing the Observable.
    rank: int  # Derivative rank of the Observable.

    def __repr__(self):
        return self.string

    # For sorting: convention is in ascending order of name

    def __lt__(self, other):
        if not isinstance(other, Observable):
            raise TypeError("Second argument is not an observable.")
        return self.string < other.string

    # TODO: This may be redundant. I believe python does this proccess internally.
    def __gt__(self, other):
        if not isinstance(other, Observable):
            raise TypeError("Second argument is not an observable.")
        return other.__lt__(self)

    def __eq__(self, other):
        if not isinstance(other, Observable):
            raise TypeError("Second argument is not an observable.")
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


def labels_to_index_list(labels, n):  # n = number of observables
    index_list = [list() for _ in range(2 * n)]
    for key in sorted(labels.keys()):
        for a in labels[key]:
            index_list[a].append(key)
    return index_list


def index_list_to_labels(index_list):
    labels = dict()
    for i, li in enumerate(index_list):
        for ind in li:
            if ind in labels.keys():
                labels[ind].append(i)
            else:
                labels[ind] = [i]
    return labels