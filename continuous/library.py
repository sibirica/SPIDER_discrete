import copy
from functools import reduce
from operator import add
from typing import List, Dict, Union, Iterable, Any, Optional, Iterator
from numbers import Real
import numpy as np
from warnings import warn
from numpy import inf

from commons.library import *


# noinspection PyArgumentList
@dataclass
class LibraryPrimitive(object):
    """
    Object representing a library primitive. Stores the primitive's derivative order, corresponding Observable, rank and
    complexity. The term 'observable' usually refers to this class (or to IndexedPrimitive) rather than an
    Observable object.

    :attribute dorder: DerivativeOrder object representing the primitive's spatial and temporal derivative orders.
    :attribute observable: Observable object storing the observable being referenced by the primitive.
    :attribute rank: Tensor rank of the primitive.
    :attribute complexity: Complexity score of the primitive.
    """
    dorder: DerivativeOrder = None
    observable: Observable = None
    rank: int = field(init=False)
    complexity: int = field(init=False)

    def __post_init__(self):
        self.rank = self.dorder.xorder + self.observable.rank
        self.complexity = self.dorder.complexity + 1

    def __repr__(self) -> str:
        tstring, xstring = create_derivative_string(self.dorder.torder, self.dorder.xorder)
        return f'{tstring}{xstring}{self.observable}'

    # For sorting: convention is (1) in ascending order of name/observable, (2) in *ascending* order of dorder

    def __lt__(self, other: 'LibraryPrimitive') -> bool:
        """
        Explicitly defines the < (lesser than) operator between two LibraryPrimitive objects.
        If two LibraryPrimitive have the same observable, self < other iff self.dorder < other.dorder. Else, compare the
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

    def __gt__(self, other: 'LibraryPrimitive') -> bool:
        """
        Explicity defines the equivalence a < b <-> b > a, when a and b are LibraryPrimitive objects.

        :param other: LibraryPrimitive to compare to.
        :return: Test comparison result.
        """
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        return other.__lt__(self)

    def __eq__(self, other: 'LibraryPrimitive') -> bool:
        """
        Explicitly defines the == (equals) operation between two LibraryPrimitive objects.
        Two LibraryPrimitive objects are deemed equal if they have the same observable and derivative order.

        :param other: LibraryPrimitive to compare to.
        :return: Inequality boolean result.
        """
        if not isinstance(other, LibraryPrimitive):
            raise TypeError("Second argument is not a LibraryPrimitive.")
        return self.observable == other.observable and self.dorder == other.dorder

    def __ne__(self, other: 'LibraryPrimitive') -> bool:
        """
        Explicity defines != as the negation of == for LibraryPrimitive objects.

        :param other: LibraryPrimitive to compare to.
        :return: Test comparison result.
        """
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
    Object representing an IndexedPrimitive. For example the x component of a vector quantity. The term 'observable'
    usually refers to this class (or to LibraryPrimitive) rather than an Observable object.
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
        Initializes an IndexedPrimitive given a LibraryPrimitive or another IndexedPrimitive. Two constructors are
        available. The normal constructor initializes the object from a LibraryPrimitive along with the space_orders and
        obs_dim arguments. The modifying constructor creates a copy of another IndexedPrimitive with its dimensional
        orders attribute modified as per the newords argument.
        NOTE: Attempting to initialize the object with more information than the necessary for exactly one constructor
        will raise a TypeError.

        :param prim: Primitive to which initialize the class. It may be a LibraryPrimitive or an IndexedPrimitive.
        :param space_orders: List containing the order of the spatial derivatives. Ex: [1,2,3] would represent a first
        order derivative in x, a second order derivative in y, and a third order derivative in z. Only applied when
        initializng from a LibraryPrimitve.
        :param obs_dim: Integer representing the Dimension of the observable/tensor. For example the x component of a
        velocity field would have this value set to 0. Only applied when initializng from a LibraryPrimitve.
        :param newords: (New orders) List containing the order of the spatial and time derivatives. Ex: [1,2,3,0] would
        represent afirst order derivative in x, a second order derivative in y, a third order derivative in z, and no
        time derivatives. Only used when initialized from another IndexedPrimitive.
        """

        # Assert exactly one of the constructors is given.
        # https://drive.google.com/file/d/16HE6t1pO0l4yoS-l_Mhrme3lPT9T1_7R/view?usp=sharing
        # Update Note: obs_dim may be none if the primitive is scalar.
        if (newords is not None) ^ (space_orders is None):
            raise TypeError(f"IndexedPrimitive must be initialized with newords, XOR obs_dim and space_orders.")

        self.dorder = prim.dorder  # DerivativeOrder object representing time and space derivative orders of prim.
        self.observable = prim.observable
        self.rank = prim.rank
        self.complexity = prim.complexity
        if newords is None:  # normal constructor
            """
            Normal constructor from an LibraryPrimitive. The constructed IndexedPrimitive has the same properties as the
            base LibraryPrimitive, its dimorders is the concatenation of the space and temporal orders, and its obs_dim
            is taken as a __init__ argument.
            """

            # Asserts the prim is of type LibraryPrimitive
            if not isinstance(prim, LibraryPrimitive):
                raise TypeError(f"The IndexedPrimitive normal (obs_dim and space_orders) constructor requires "
                                f"prim to be a LibraryPrimitive, a {type(prim)} was provided instead.")

            self.dimorders = space_orders + [self.dorder.torder]
            self.obs_dim = obs_dim
        else:
            """
            Modifying constructor from an IndexedPrimitive. The constructed object has the same properties as the base
            primitive, but its dimorders are modifyied to be the same as the newords __init__ argument.
            """
            # Asserts the prim is of type IndexedPrimitive.
            if not isinstance(prim, type(self)):
                raise TypeError(f"The IndexedPrimitive neword constructor requires prim to be a IndexedPrimitive, a "
                                f"{type(prim)} was provided instead.")
            self.dimorders = newords
            self.obs_dim = prim.obs_dim
        self.ndims = len(self.dimorders)
        self.nderivs = sum(self.dimorders)

    def __repr__(self) -> str:
        """
        IndexedPrimitives are represented as 'dx^i dy^j dz^k dt^l O_e' where O stands for the observable, and e is the
        dimension subscript (Ex. v_x is the x component of v). If the derivative orders (a.k.a. i,j,k,l) are 1, they are
        ommited, if they are 0 the whole derivative term is ommited.
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

    def __eq__(self, other: 'IndexedPrimitive') -> bool:
        """
        Two IndexedPrimitive are deemed equal if they have the same dimension order, observable being represented, and
        indexed dimension (e.g. they both correspond to the x component).
        """
        return (self.dimorders == other.dimorders and self.observable == other.observable
                and self.obs_dim == other.obs_dim)

    def __mul__(self, other: Union['IndexedPrimitive', 'IndexedTerm']) -> 'IndexedTerm':
        """
        Defines multiplication between IndexedPrimitive objects, and between an IndexedTerm and an IndexedPrimitive.

        Multiplication between IndexedPrimitive objects returns an IndexedTerm constructed from a list containing both
        IndexedPrimitive objects in the order provided.

        Multiplication between an IndexedTerm and an IndexedPrimitive returns a copy of the IndexedTerm with the
        IndexedPrimitive appended to the former's obs_list.

        :param other: IndexedTerm or IndexedPrimitive to multiply by.
        :return: IndexedTerm corresponding to the result of the multiplication.
        """
        if isinstance(other, IndexedTerm):
            return IndexedTerm(obs_list=[self] + other.obs_list)
        else:
            return IndexedTerm(obs_list=[self, other])

    def succeeds(self, other, dim: int) -> bool:
        """
        Tests if other is a derivative of self in the given dimension (dim).
        """
        try:
            copyorders = copy.deepcopy(self.dimorders)
            copyorders[dim] += 1
            return all([copyorders == other.dimorders,
                        self.observable == other.observable,
                        self.obs_dim == other.obs_dim
                        ])
        except AttributeError:
            raise TypeError(f"Can't infer if type {type(other)} succceeds a IndexedPrimitive")

    def diff(self, dim: int) -> 'IndexedPrimitive':
        """
        Returns an IndexedPrimitive with the same properties as self but with an extra derivative order in the given
        dimension (dim).

        :param dim: Dimension to take the derivative.
        :return: IndexedPrimitive with a higher derivative order.
        """
        newords = copy.deepcopy(self.dimorders)
        newords[dim] += 1
        return IndexedPrimitive(self, newords=newords)


# TODO: Move common functionality to commons
# noinspection DuplicatedCode
class LibraryTensor(object):
    """
    Unindexed version of LibraryTerm. May also store products of observables as a list of LibraryPrimitive objects.

    :attribute obs_list: List of observables represented as a list of LibraryPrimitive objects.
    :attribute rank: Tensor rank of the object.
    :attribute complexity: Complexity score of the object.
    """

    def __init__(self, observables: Union[LibraryPrimitive, List[LibraryPrimitive]]):
        """
        Constructs a LibraryTensor given a LibraryPrimitive or a list of LibraryPrimitive objects.

        :param observables: LibraryPrimitive or a list of LibraryPrimitive objects to act as base for the constructor.
        """
        # constructor for library terms consisting of a primitive with some derivatives
        if isinstance(observables,
                      LibraryPrimitive):
            self.obs_list = [observables]
        else:  # constructor for library terms consisting of a product
            self.obs_list = observables
        self.rank = sum([obs.rank for obs in self.obs_list])
        self.complexity = sum([obs.complexity for obs in self.obs_list])

    def __mul__(self, other: Union[int, 'LibraryTensor']) -> 'LibraryTensor':
        """
        Multiplying LibraryTensor is defined as combining their observable (LibraryPrimitive) lists.

        :param other: LibraryTensor to be multiplied by.
        :return: LibraryTensor resulting from the product.
        """
        if isinstance(other, LibraryTensor):
            return LibraryTensor(self.obs_list + other.obs_list)
        elif isinstance(other, int):
            if other == 1:
                return self
            raise ValueError(f"Cannot multiply {type(self)} by integers other than 1")
        else:
            raise TypeError(f"Cannot multiply {type(self)}, {type(other)}")

    def __rmul__(self, other) -> Union['LibraryTensor', Any]:
        """
        Establishes LibraryTensor multiplication as commutative, with 1 as identity.
        """
        if other != 1:
            return other.__mul__(self)
        else:
            return self

    def __repr__(self) -> str:
        """
        LibraryTensors are represented as the string representation of their observables (LibraryPrimitive) joined by
        a spaced asterisk (multiplication sign) ' * '. E.g. 'dx^2 f * dt g'

        :return: String representation of this LibraryTensor.
        """
        repstr = [str(obs) + ' * ' for obs in self.obs_list]
        return reduce(add, repstr)[:-3]


class LibraryTerm(object):
    """
    Represents a single library term, which consists of differential operators, observables, their order, alongside
    with any free and summation indexes.
    NOTE: be careful not to modify index_list or labels without remaking because the references are reused!

    :attribute canon_dict: Stores ambiguous canonicalizations (which shouldn't exist for less than 6 indices). This
    dictionary is common to ALL LibraryTerm objects.
    :attribute obs_list: List of observables represented as a list of LibraryPrimitive objects.
    :attribute libtensor: LibraryTensor object storing observables.
    :attribute rank: Tensor rank of the object. In this system we only have rank 0 and 1 tensor. Rank changes after
    contraction.
    :attribute complexity: Complexity score of the object.
    :attribute labels: The indexes of the Observables and its differentials represented as a Dict[int, List[int]].
    :attribute index_list: The indexes of the Observables and its differentials represented as a List[List[int]].
    https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels
    :attribute der_index_list: Similar to index_list, but only contains derivative indexes.
    :attribute obs_index_list: Similar to index_list, but only contains observable indexes.
    :attribute is_canonical: Flags wether this object is in canonical form.
    """
    canon_dict = dict()  # used to store ambiguous canonicalizations (which shouldn't exist for less than 6 indices)

    def __init__(self, libtensor: LibraryTensor,
                 labels: Dict[int, Union[List[int], Tuple[int]]] = None,
                 index_list: List[List[int]] = None):
        """
        Initializes a LibraryTerm by wrapping a LibraryTensor with a `labels` dict XOR an index_list. For more
        information on labels and index_list objects please refer to
        https://github.com/sibirica/SPIDER_discrete/wiki/Index-Lists-and-Labels
        NOTE: The constructor takes exactly one constructor argument, providing two raises a TypeError.

        :param libtensor: LibraryTensor to serve as base for the LibraryTerm.
        :param labels: The indexes of the Observables and its differentials represented as a Dict[int, List[int]].
        :param index_list: The indexes of the Observables and its differentials represented as a List[List[int]].
        """

        # Assert exactly one of the constructors is given.
        # https://drive.google.com/file/d/16HE6t1pO0l4yoS-l_Mhrme3lPT9T1_7R/view?usp=sharing
        if (labels is not None) ^ (index_list is None):
            raise TypeError("LibraryTerm must be initialized with a labels dictionary XOR an index_list.")

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

    def __add__(self, other: Union['LibraryTerm', 'Equation']) -> 'TermSum':
        """
        Defines addition between two LibraryTerm objects and between a LibraryTerm and an Equation (or TermSum). All
        cases result in a TermSum consisting of the concatenation of all LibraryTerm objects contained in both objects
        being summed.

        :param other: LibraryTerm, Equation, or TermSum to add to self.
        :return: TermSum corresponding to the operation result.
        """
        if isinstance(other, LibraryTerm):
            return TermSum([self, other])
        else:
            return TermSum([self] + other.term_list)

    def __eq__(self, other) -> bool:
        """
        Explicitly defines the == (equals) operation between two LibraryTerm objects.
        Two LibraryTerm objects are deemed equal if they have the same observable (LibraryPrimitive) list (order
        sensitive) and index_list.
        NOTE: assumes both LibraryTerm are in canonical form.

        :param other: LibraryTerm to compare equality with.
        :return: Comparison test result.
        """
        if isinstance(other, LibraryTerm):
            return self.obs_list == other.obs_list and self.index_list == other.index_list
        else:
            return False

    def __ne__(self, other) -> bool:
        """
        Explicity defines the != (not equals) operator as the negation of the == (equals) operator for LibraryTerms.
        NOTE: assumes both LibraryTerm are in canonical form.

        :param other: LibraryTerm to compare inequality equality with.
        :return: Comparison test result.
        """
        return not self.__eq__(other)

    def __lt__(self, other) -> bool:
        """
        Explicity defines the < (lesser than) operator between a LibraryTensor and another object with string
        representation. The comparison result is the same as if it was applied to the objects' string representation.

        :param other: Object to compare inequality with.
        :return: Comparison test results.
        """
        return str(self) < str(other)

    def __gt__(self, other) -> bool:
        """
        Explicity defines the > (greater than) operator between a LibraryTensor and another object with string
        representation. The comparison result is the same as if it was applied to the objects' string representation.

        :param other: Object to compare inequality with.
        :return: Comparison test results.
        """
        return str(self) > str(other)

    def __repr__(self) -> str:
        """
        Defines the representation of a LibraryTerm as a concatenation of indexed LibraryPrimitive representations
        joined by ` * `. For more details see the label_repr() function.

        :return: String representation of self.
        """
        repstr = [label_repr(obs, ind1, ind2) + ' * ' for (obs, ind1, ind2) in
                  zip(self.obs_list, num_to_let(self.der_index_list), num_to_let(self.obs_index_list))]
        return reduce(add, repstr)[:-3]

    def __hash__(self) -> int:  # it's nice to be able to use LibraryTerms in sets or dicts
        """
        A Library Term's hash is the hash of its string representation.
        NOTE: Comparing LibraryTerms with hashes is only conclusive when applied between the hashes generated from
        canonical LibraryTerms.

        :return: The Library Term's hash.
        """
        return hash(self.__repr__())

    # TODO: Move common functionality to commons.
    # noinspection DuplicatedCode
    def __mul__(self, other) -> Union['LibraryTerm', 'Equation']:
        """
        This method explicity defines the * (multiplication) operator for LibraryTerm, when the object is on the left of
        the operand.
        The multiplication operation can be thought of as the scalar product. Multiplying a scalar by a scalar yields a
        scalar, a scalar by a vector yield a vector, and a vector by a vector yields a scalar.
        NOTE: The product is always returned in canonical form.

        :param other: Object to be multiplied by.
        :return: Product of self*other.
        """
        if isinstance(other, LibraryTerm):
            if self.rank < other.rank:
                # Multiplying by scalars is less computationally intensive.
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
            # Defines 1 as the identity of multiplication operation in LibraryTerms
            return self
        elif isinstance(other, Equation):
            return other.__mul__(self)
        else:
            raise TypeError(f"Cannot multiply {type(self)}, {type(other)}")

    def __rmul__(self, other):
        """
        Explicity defines multiplication as commutative.

        :param other: Object to be multiplied by.
        :return: Product.
        """
        return self.__mul__(other)

    # TODO: Move common functionality to commons.
    # noinspection DuplicatedCode
    def structure_canonicalize(self) -> 'LibraryTerm':
        """
        Reorders the object's obs_list so that the observables are stored in ascending order. See
        LibraryPrimitive.__lt__() for more details of evaluating inequalities.

        :return: LibraryTerm with similar properties, but with observables stored in ascending order.
        """
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

    def index_canonicalize(self) -> 'LibraryTerm':
        """
        Compares the current indexation of the object with their canonical form. If the indexes are already in canonical
        order, returns self; else, returns a new LibraryTerm with similar properties but with canonicalized indices.

        :return: A LibraryTerm with canonical indices, but unchanged properties.
        """
        # inc = 0
        # if len(self.labels[0])==2: # if multiple i's, need to increment all indices
        #    inc = 1
        subs_dict = canonicalize_indices(flatten(self.index_list))
        new_index_list = [[subs_dict[i] for i in li] for li in self.index_list]
        if all([li1 == li2 for li1, li2 in zip(self.index_list, new_index_list)]):  # no changes were made
            return self
        return LibraryTerm(self.libtensor, index_list=new_index_list)

    # TODO: Move common functionality to commons.
    # noinspection DuplicatedCode
    def reorder(self, template: Iterable) -> 'LibraryTerm':
        """
        Reorders the objects' observable list and indexes as to follow an order indicated by a template.
        Ex: obs_list = [u, v, p, w], template = [13, 2, -5, 3] -> obs_list = [p, v, w, u]

        :param template: An iterable whose items can be sorted (support the < operator).
        :return: A LibraryTerm with reordered objects and indexes, but unchanged properties.
        """
        indexed_zip = zip(self.obs_list, self.der_index_list, self.obs_index_list, template)
        sorted_zip = sorted(indexed_zip, key=lambda x: x[3])
        sorted_obs = [e[0] for e in sorted_zip]
        sorted_ind1 = [e[1] for e in sorted_zip]
        sorted_ind2 = [e[2] for e in sorted_zip]
        sorted_ind = flatten(list(zip(sorted_ind1, sorted_ind2)))
        sorted_libtens = LibraryTensor(sorted_obs)
        return LibraryTerm(sorted_libtens, index_list=sorted_ind)

    # TODO: Move common functionality to commons.
    # noinspection DuplicatedCode
    def canonicalize(self) -> 'LibraryTerm':
        """
        Returns the LibraryTerm's canonical representation and set LibraryTerm.is_canonical to True (used to determine
        if the LibraryTerm is valid). Other than ordering and the aforementioned is_canonical flag, the objects'
        properties remain unchaged.

        :return: LibraryTerm's canonical representation.
        """
        if self.is_canonical:
            return self
        str_canon = self.structure_canonicalize()
        if str_canon in self.canon_dict:
            canon = self.canon_dict[str_canon]
            self.is_canonical = (self == canon)
            canon.is_canonical = True
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
        self.is_canonical = (self == canon)
        for alt_canon in alternative_canons:
            self.canon_dict[alt_canon] = canon
        canon.is_canonical = True
        return canon

    def increment_indices(self, inc: int) -> 'LibraryTerm':
        """
        Generates a new LibraryTerm identical to self, but whose indices were incremented by a fixed integer amount inc.

        :param inc: Integer amount to increment indices by.
        :return: LibraryTerm with similar propreties, but whose indices were incremented by inc.
        """
        index_list = [[index + inc for index in li] for li in self.index_list]
        return LibraryTerm(LibraryTensor(self.obs_list), index_list=index_list)

    # TODO: Move common functionality to commons.
    # noinspection DuplicatedCode
    def dt(self) -> 'Equation':
        """
        Returns the time derivative of the LibraryTerm. The returned object is of type Equation and is in canonical
        form.

        :return: Equation representing the time derivative of the LibraryTerm.
        """
        terms = []
        for i, obs in enumerate(self.obs_list):  # Product rule
            new_obs = obs.dt()
            # note: no need to recanonicalize terms after a dt
            lt = LibraryTerm(LibraryTensor(self.obs_list[:i] + [new_obs] + self.obs_list[i + 1:]),
                             index_list=self.index_list)
            terms.append(lt)
        ts = TermSum(terms)
        return ts.canonicalize()

    # TODO: Move common functionality to commons.
    # noinspection DuplicatedCode
    def dx(self) -> 'Equation':
        """
        Returns the spatial derivative of the LibraryTerm. The returned object is of type Equation and is in canonical
        form.
        NOTE: Assumes that the index `i` always denotes the free index. Using `i` for repeated indexes will result in
        equations with terms ill-defined under Einstein's convention (e.g. v_i * di v_i).

        :return: Equation representing the spatial derivative of the LibraryTerm.
        """
        terms = []
        for i, obs in enumerate(self.obs_list):  # Product rule
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


class IndexedTerm(object):  # LibraryTerm with i's mapped to x/y/z
    """
    LibraryTerm with i's mapped to x/y/z. Equivalent to LibraryTerm, but stores observables as IndexedPrimtives rather
    than LibraryPrimitives.

    :attribute complexity: Complexity score of the object. Equals the sum of the complexity score of the stored
    IndexedPrimitive objects in obs_list.
    :attribute nderivs: Highest sum of derivative order across all IndexedPrimitive objects stored in obs_list.
    :attribute ndims: Number of spatial-temporal dimensions of the stored observables.
    :attribute obs_list: List of observables (IndexedPrimitive).
    :attribute rank: Tensor rank of the stored observables.
    """

    def __init__(self, libterm: LibraryTerm = None,
                 space_orders: List[List[int]] = None,
                 obs_dims: List[Union[int, None]] = None,
                 obs_list: List[IndexedPrimitive] = None):
        """
        Initializes an IndexedTerm object given a list of IndexedPrimitives, OR a LibraryTerm, list of observable
        dimension numbers, and list of space orders.
        NOTE: The initialization takes exactly the information necessary for one constructor, providing more or less
        will raise a TypeError.

        :param libterm: LibraryTerm to serve as constructor base.
        :param space_orders: List containing the space_orders of each IndexedPrimitive to be stored in self.obs_list.
        :param obs_dims: List containing the obs_dim of each IndexedPrimitive to be stored in self.obs_list.
        :param obs_list: List of IndexedPrimitive to initialize the object with.
        """

        # Asserts exactly one of the costructors is given.
        # https://drive.google.com/file/d/16HE6t1pO0l4yoS-l_Mhrme3lPT9T1_7R/view?usp=sharing
        if (obs_list is not None) ^ (libterm is None or space_orders is None or obs_dims is None):
            raise TypeError("Indexed term must be initialized with a obs_list, XOR libterm and space_orders and "
                            "obs_dims.")

        if obs_list is None:  # Initializes from scratch given a LibraryTerm, space_orders, and obs_dim.
            self.rank = libterm.rank
            self.complexity = libterm.complexity
            # self.obs_dims = obs_dims
            nterms = len(libterm.obs_list)

            # Turns the base LibraryTerm.obs_list from a List[LibraryPrimitive] into a List[IndexedPrimitive].
            self.obs_list = copy.deepcopy(libterm.obs_list)
            for i, obs, sp_ord, obs_dim in zip(range(nterms), libterm.obs_list, space_orders, obs_dims):
                self.obs_list[i]: IndexedPrimitive = IndexedPrimitive(obs, sp_ord, obs_dim)
            self.obs_list: List[IndexedPrimitive]  # This line is for type hinting only.

            self.ndims = len(space_orders[0]) + 1
            self.nderivs = np.max([p.nderivs for p in self.obs_list])
        else:  # Initializes directly from an observable list, i.e. List[IndexedPrimitive]
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

    def __repr__(self) -> str:
        """
        Defines the representation of an IndexedTerm as the string representation of the IndexedPrimitives stored in its
        obs_list joined by ' * '.

        :return: Representation of this IndexedTerm.
        """
        repstr = [str(obs) + ' * ' for obs in self.obs_list]
        return reduce(add, repstr)[:-3]

    def __mul__(self, other: Union['IndexedTerm', IndexedPrimitive]) -> 'IndexedTerm':
        """
        Defines multiplication between IndexedTerm objects, and between an IndexedTerm and an IndexedPrimitive.
        Multiplication between IndexedTerm objects returns an IndexedTerm constructed from the concatenation of their
        obs_list.

        Multiplication between an IndexedTerm and an IndexedPrimitive returns a copy of the IndexedTerm with the
        IndexedPrimitive appended to the former's obs_list.

        :param other: IndexedTerm or IndexedPrimitive to multiply by.
        :return: IndexedTerm corresponding to the result of the multiplication.
        """
        if isinstance(other, IndexedTerm):
            return IndexedTerm(obs_list=self.obs_list + other.obs_list)
        else:
            return IndexedTerm(obs_list=self.obs_list + [other])

    def drop(self, obs: Union[IndexedPrimitive, Any]) -> 'IndexedTerm':
        """
        Returns a copy of this IndexedTerm whithout the first instance of provided IndexedPrimitive in its object list.
        NOTE: Does not check if the IndexedPrimitive is present in self.obs_list if len(self.obs_list) <= 1. Simply
        returning the equivalent of a ConstantTerm.

        :param obs: Observable to be dropped.
        :return: A copy of self without the first instance of the given observable.
        """
        # print(self.obs_list)
        obs_list_copy = copy.deepcopy(self.obs_list)
        if len(obs_list_copy) > 1:
            obs_list_copy.remove(obs)
        else:
            obs_list_copy = []
        return IndexedTerm(obs_list=obs_list_copy)

    def drop_all(self, obs: Union[IndexedPrimitive, Any]) -> 'IndexedTerm':  # remove *ALL* instances of obs
        """
        Returns a copy of this IndexedTerm whithout any instances of the provided IndexedPrimitive in its object list.
        NOTE: Does not check if the IndexedPrimitive is present in self.obs_list if len(self.obs_list) <= 1. Simply
        returning the equivalent of a ConstantTerm.

        :param obs: Observable to be dropped.
        :return: A copy of self without any instances of the given observable.
        """
        if len(self.obs_list) > 1:
            obs_list_copy = list(filter(obs.__ne__, self.obs_list))
        else:
            obs_list_copy = []
        return IndexedTerm(obs_list=obs_list_copy)

    def diff(self, dim: int) -> Generator['IndexedTerm', None, None]:
        """
        Differentiates the IndexedTerm in respect to dim. Creates a generator that yields each term of the product rule
        sum one at a time.

        :param dim: Dimension to differentiate by.
        :return: Generator of IndexedTerm objects representing the sum terms of the product rule.
        """
        for i, obs in enumerate(self.obs_list):
            yield obs.diff(dim) * self.drop(obs)


class ConstantTerm(IndexedTerm):
    """
    An IndexTerm with an empty obs_list.
    NOTE: must be handled separately in derivatives.

    :attribute complexity: Complexity score of the object. Equals the sum of the complexity score of the stored
    IndexedPrimitive objects in obs_list.
    :attribute nderivs: Highest sum of derivative order across all IndexedPrimitive objects stored in obs_list.
    :attribute ndims: Number of spatial-temporal dimensions of the stored observables.
    :attribute obs_list: List of observables (IndexedPrimitive).
    :attribute rank: Tensor rank of the stored observables.
    """

    def __init__(self):
        super().__init__(obs_list=[])
        self.rank = 0
        self.is_canonical = True

    def __repr__(self) -> str:
        return "1"

    @staticmethod
    def dt() -> None:
        """
        The derivative of a constant is zero.

        :return: None
        """
        return None

    @staticmethod
    def dx() -> None:
        """
        The derivative of a constant is zero.

        :return: None
        """
        return None


def label_repr(prim: LibraryPrimitive, ind1: List[str], ind2: List[str]) -> str:
    """
    Given a LibraryPrimitive, the list of differencial indexes, and its list of obsevable indexes. Returns a formatted
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


# noinspection PyArgumentList
def raw_library_tensors(observables: Union[List[Observable], Tuple[Observable]],
                        obs_orders: Union[List[int], Tuple[list]],
                        nt: int,
                        nx: int,
                        max_order: DerivativeOrder = DerivativeOrder(inf, inf),
                        zeroidx: int = 0) -> Generator[Union[LibraryTensor, int], None, None]:
    """
    Instatiates a Generator that will yield all LibraryTensors that can be constructed using the Observable objects
    specified by 'observables', 'obs_order', and 'zeroidx', that contain 'nt' or less time derivatives, 'nx' or less
    spatial derivatives, and whose DerivativeOrder is smaller than 'max_order'.
    NOTE: For exclusive use of generate_terms_to.

    :param observables: List of Observable object to generate tensors from.
    :param obs_orders: List of integers representing the order of each Observable, i.e. how many times each term
    appears.
    :param nt: Number of times a time derivative takes place.
    :param nx: Number of times a spacial derivative takes place.
    :param max_order: Maximum derivative order of a tensor.
    :param zeroidx: Index from which start to looking for non-zero terms in obs_order.
    :return: Generator that yields all LibraryTensor objects that can be constructed from the function's arguments.
    """
    # print(obs_orders, nt, nx, max_order)
    while obs_orders[zeroidx] == 0:  # Finds the first observable whose order is not zero. Set its index to zeroidx.
        zeroidx += 1
        if zeroidx == len(observables):  # If all indexes are zero, yield a constant term.
            yield 1
            return
    if sum(obs_orders) == 1:  # Checks for terms with only one observable.
        i = obs_orders.index(1)
        do = DerivativeOrder(nt, nx)
        if max_order >= do:  # Only yields a tensor if its derivative orders are smaller than max_order.
            prim = LibraryPrimitive(do, observables[i])
            yield LibraryTensor(prim)
        return
    for i in range(nt + 1):
        for j in range(nx + 1):
            do = DerivativeOrder(i, j)
            if max_order >= do:  # Only yields a tensor if its derivative orders are smaller than max_order.
                prim = LibraryPrimitive(do, observables[zeroidx])
                term1 = LibraryTensor(prim)
                new_orders = list(obs_orders)
                new_orders[zeroidx] -= 1
                if obs_orders[zeroidx] == 1:  # reset max_order since we are going to next terms
                    do = DerivativeOrder(inf, inf)
                for term2 in raw_library_tensors(observables, new_orders, nt - i, nx - j, do, zeroidx):
                    yield term2 * term1  # reverse back to ascending order here


def generate_terms_to(order: int,
                      observables: List[Observable] = None,
                      max_observables: int = 999) -> List[Union[ConstantTerm, LibraryTerm]]:
    """
    Given a list of Observable objects and a complexity order, returns the list of all LibraryTerms resulting from the
    manipulation of these Observable objects whose complexity is equal to or smaller than order, and that contain no
    more than max_observable Observable objects, also includes a ConstantTerm.

    :param order: Max complexity order that terms will be generated to.
    :param observables: list of Observable objects used to construct the terms.
    :param max_observables: Maximum number of Observables (and derivatives) in a single term.
    :return: List of all possible LibraryTerms whose complexity is less than or equal to order, that can be generated
    using the given observables.
    """
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


class Equation(object):  # can represent equation (expression = 0) OR expression
    """
    Equation objects may represent an equation (an expression that equals to zero) or an expresion.

    :attribute coeffs: Sequence of Real numbers that represent the coefficient of each term in the Equation.
    :attribute complexity: Complexity score of the object. Equal to the sum of the complexity attribute of each object
    stored in self.term_list.
    :attribute rank: Tensor rank of the equation or expression.
    :attribute term_list: Sequence of LibraryTerm objects representing the tersm of the Equation.
    """

    # terms are LibraryTerms, coeffs are real numbers
    def __init__(self, term_list: Union[List[LibraryTerm], Tuple[LibraryTerm]],
                 coeffs: Union[List[Real], Tuple[Real]]):
        """
        Constructs an Equation given a sequence of LibraryTerm objects and the list of associated coefficients.

        :param term_list: Sequence of LibraryTerm objects that will represent the terms of the Equation.
        :param coeffs: Sequence of floats that will represent the coefficient of each term in the Equation.
        """
        if len(term_list) != len(coeffs):
            warn(RuntimeWarning("Initializing Equation with term_list and coeffs with unequal length. This might "
                                "lead to undesired/unpredictable behaviour!"))
        content = zip(term_list, coeffs)
        sorted_content = sorted(content, key=lambda x: x[0])
        # note that sorting guarantees canonicalization in equation term order
        self.term_list = [e[0] for e in sorted_content]
        self.coeffs = [e[1] for e in sorted_content]
        self.rank = term_list[0].rank
        self.complexity = sum([term.complexity for term in term_list])  # another choice is simply the number of terms

    def __add__(self, other: 'Equation') -> 'Equation':
        """
        Summing two Equation objects results in another Equation object that concatenates the base objects' term_list
        and coeffs.

        :param other: Equation to add to self.
        :return: Sum of both Equation objects.
        """
        if isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise TypeError(f"Cannot add object of type {type(other)} with object of type {type(self)}.\n"
                            f"Second argument ({other}) is not an Equation!")

    def __rmul__(self, other) -> Union['Equation', None]:
        """
        Explicitly defines multiplication for Equation objects. Multiplying an Equation by a LibraryTerm results in
        an Equation where every term of the original Equation is multiplied by the LibraryTerm and is then
        canonicalized. Multiplying an Equation by a number simply returns a similar Equation where each item of
        self.coeffs is multiplyied by that number.

        :param other: Object to multiply the Equation by.
        :return: Equation resulting of the product other*self.
        """
        if isinstance(other, LibraryTerm):
            return Equation([(other * term).canonicalize() for term in self.term_list], self.coeffs)
        elif isinstance(other, (float, int)):  # multiplication by number
            if other == 1:
                return self
            elif other == 0:
                return None
            else:
                return Equation(self.term_list, [other * c for c in self.coeffs])
        elif isinstance(other, ConstantTerm):
            return self
        else:
            raise TypeError(f'Cannot multiply type {type(other)} with type {type(self)}.')

    def __mul__(self, other) -> 'Equation':
        """
        Explicity establishes the commutativity of multiplication with Equation objects.

        :param other: Object to multiply the Equation by.
        :return: Equation resulting of the product other*self.
        """
        return self.__rmul__(other)

    def __repr__(self) -> str:
        """
        An Equation is represented by "{coeff} * term" iterated over self.term_list and self.coeffs.
        E.g.: 0.9999999997877309 * di p + 5.39269567032929e-11 * p * di p + 1.0 * u_j * dj u_i

        :return: Representation of the Equation.
        """
        repstr = [str(coeff) + ' * ' + str(term) + ' + ' for term, coeff in zip(self.term_list, self.coeffs)]
        return reduce(add, repstr)[:-3]

    def __str__(self) -> str:
        """
        Transforming an Equation into a string is equivalent to getting its representation and appending " = 0".

        :return: String of the Equation.
        """
        return self.__repr__() + " = 0"

    def __eq__(self, other: 'Equation') -> bool:
        """
        For two Equation objects to be deemed equal they must have the same term_list and the same coeffs.

        :param other: Equation to compare equality two.
        :return: Comparison result.
        """
        if isinstance(other, Equation):
            return self.term_list == other.term_list and self.coeffs == other.coeffs
        else:
            raise TypeError(f'Cannot compare equality between type {type(other)} and type {type(self)}.')

    def dt(self) -> Union['Equation', 'TermSum', None]:
        """
        Takes the temporal derivative of self, does so by taking the time derivative of every element of term_list,
        except ConstantTerm objects, then canonicalizing. If all objects in term_list are type ConstantTerm returns
        None.

        :return: The first time derivative of self. Treating 0 as None.
        """
        components = [coeff * term.dt() for term, coeff in zip(self.term_list, self.coeffs)
                      if not isinstance(term, ConstantTerm)]
        if not components:
            return None
        return reduce(add, components).canonicalize()

    def dx(self) -> Union['Equation', 'TermSum', None]:
        """
        Takes the spatial derivative of self, does so by taking the space derivative of every element of term_list,
        except ConstantTerm objects, then canonicalizing. If all objects in term_list are type ConstantTerm returns
        None.

        :return: The first space derivative of self. Treating 0 as None.
        """
        components = [coeff * term.dx() for term, coeff in zip(self.term_list, self.coeffs)
                      if not isinstance(term, ConstantTerm)]
        if not components:
            return None
        return reduce(add, components).canonicalize()

    def canonicalize(self) -> 'Equation':
        """
        Canonicalizes self by combining like terms, i.e. deleting duplicates on self.term_list and adding the
        duplicate's coefficients.

        :return: Canonical version of self.
        """
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

    def eliminate_complex_term(self, return_normalization: bool = False) \
            -> Union[Tuple[LibraryTerm, Optional['Equation'], Real],
                     Tuple[LibraryTerm, Optional['Equation']]]:
        """
        Identifies and removes the term with the highest complexity score from the Equation, then normalizes the
        remaining terms by -1 * coefficient of the highest complexity term. Returns the results as a tuple with the
        first element being the LibraryTerm of the highest complexity term, the second element being the new normalized
        Equation, and (if return_normalization is True) the third element being the coefficient of the highest
        complexity term.
        NOTE: Mathematically, this is equivalent to starting with the statement Equation = 0, then moving all terms,
        except the one with the highest complexity, to the right hand side, then dividing both sides to assure the
        right hand side has coefficient 1.

        :param return_normalization: Wether to return the coefficient of the most complex term.
        :return: Tuple with the most complex LibraryTerm, Equation with the remaining terms normalized by -1 * its
        coefficient, and (if return_normalization is True) the coefficient itself.
        """
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

    def to_term(self) -> LibraryTerm:
        """
        If the Equation has a single term, returns that LibraryTerm. Raises ValueError otherwise.

        :return: The term of single term Equations.
        """
        if len(self.term_list) != 1:
            raise ValueError("Equation contains more than one distinct term")
        else:
            return self.term_list[0]


class TermSum(Equation):
    """
    Special instance of Equation where all coefficients are equal to one.

    :attribute coeffs: Sequence of Real numbers that represent the coefficient of each term in the Equation.
    :attribute complexity: Complexity score of the object. Equal to the sum of the complexity attribute of each object
    stored in self.term_list.
    :attribute rank: Tensor rank of the equation or expression.
    :attribute term_list: Sequence of LibraryTerm objects representing the tersm of the Equation.
    """

    # terms are LibraryTerms, coeffs are real numbers
    def __init__(self, term_list: Union[List[LibraryTerm], Tuple[LibraryTerm]]):
        """
        Constructs a TermSum given a sequence of LibraryTerm objects.

        :param term_list: Sequence of LibraryTerm objects that will represent the terms of the TermSum.
        """
        super().__init__(term_list=term_list, coeffs=[1] * len(term_list))

    def __str__(self):
        """
        A TermSum 's string is constructed in the same manner as an Equation 's; however, as all coefficients are known
        to be 1 their representation is ommited.

        :return: String representing the TermSum
        """
        repstr = [str(term) + ' + ' for term in self.term_list]
        return reduce(add, repstr)[:-3]

    def __add__(self, other: Union['TermSum', Equation]) -> Union['TermSum', Equation]:
        """
        Defines addition for TermSum objects. Adding two TermSum objects results in a third TermSum object created from
        the concatenation of the base TermSum objects' term_list. Adding a TermSum and an Equation results in a new
        Equation constructed from the concatenation of the base objects' term_list and coeffs.

        :param other: Object to be added to self.
        :return: Result of the addition.
        """

        if isinstance(other, TermSum):
            return TermSum(self.term_list + other.term_list)
        elif isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise TypeError(f"Cannot multiply type {type(other)} with {type(self)}.\n"
                            f"Second argument {other}) is not an Equation.")
