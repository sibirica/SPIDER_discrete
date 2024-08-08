from typing import Any, Union
from warnings import warn

import numpy as np
from findiff import FinDiff
from library import *
from commons.weight import *
from functools import reduce
from operator import mul

# if we want to use integration domains with different sizes & spacings, it might be
# better to store that information within this object as well
class IntegrationDomain(object):
    def __init__(self, min_corner, max_corner):
        # min_corner - min coordinates in each dimension; sim. max_corner
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.shape = [max_c - min_c + 1 for (min_c, max_c) in zip(self.min_corner, self.max_corner)]
        self.times = list(range(min_corner[-1], max_corner[-1] + 1))

    def __repr__(self):
        return f"IntegrationDomain({self.min_corner}, {self.max_corner})"

    def __hash__(self):  # for use in cg_dict etc.
        # here we use that min_corner and max_corner are both lists of ints
        return hash(tuple(self.min_corner + self.max_corner))

    def distance(self, pt):
        return max(self.line_dist(coord, i) for i, coord in enumerate(pt)) # L_0 norm distance
        # return np.linalg.norm([self.line_dist(coord, dim) for i, coord in enumerate(pt)]) # L_2 norm distance

    def line_dist(self, coord, dim):
        return max(0, self.min_corner[dim] - coord, coord - self.max_corner[dim])

class Weight(object): # scalar-valued weight function
    def __init__(self, m, q, k, scale=1, dxs=None):
        self.m = m
        self.q = q
        self.k = k
        self.scale = scale
        self.ready = False
        self.weight_objs = None
        if dxs is None:
            self.dxs = [1] * len(self.m)
        else:
            self.dxs = dxs

    def make_weight_objs(self):
        self.ready = True
        self.weight_objs = [weight_1d(m, q, k, dx) for (m, q, k, dx) in zip(self.m, self.q, self.k, self.dxs)]

    def get_weight_array(self, dims):
        if not self.ready:
            self.make_weight_objs()
        weights_eval = [weight.linspace(dim)[1] for (weight, dim) in zip(self.weight_objs, dims)]
        return self.scale * reduce(lambda x, y: np.tensordot(x, y, axes=0), weights_eval)

    def increment(self, dim):  # return new weight with an extra derivative on the dim-th dimension
        knew = self.k.copy()
        knew[dim] += 1
        return Weight(self.m, self.q, knew, scale=self.scale, dxs=self.dxs)

    def __neg__(self):
        return Weight(self.m, self.q, self.k, scale=-self.scale, dxs=self.dxs)

    def __mul__(self, number):
        return Weight(self.m, self.q, self.k, scale=self.scale * number, dxs=self.dxs)

    __rmul__ = __mul__

    def __repr__(self):
        return f"Weight({self.m}, {self.q}, {self.k}, {self.scale}, {self.dxs})"

    def __hash__(self):
        return hash(self.__repr__)

class Metric(object):
    n_dimensions: int # number of dimensions of space
    tensor: np.ndarray = None

    def __post_init(self):
        if self.tensor is None:
            self.tensor = np.eye(self.n_dimensions)

    def __repr__(self):
        return f"Metric({self.tensor})"

def multi_index_iterator(n_dimensions, rank):
    a = np.arange(n_dimensions**rank).reshape((n_dimensions,)*rank)
    return np.nditer(a, flags=['multi_index'])

@dataclass
class TensorWeight: # tensor-valued weight function 
    weight_dict: dict[tuple[int], Weight] # dict mapping tuples to weight functions
    rank: int # rank of tensor/library
    n_dimensions: int # number of dimensions of data

    def __call__(self, input):
        return self.weight_dict[input]

@dataclass
class FactoredTensorWeight(TensorWeight):

    # factorization
    base_weight: Weight
    tensor: np.ndarray

    @staticmethod
    def constant_tw(base_weight, tensor): # create TensorWeight equal to constant Tensor * scalar Weight
        weight_dict = dict()
        n_dimensions, rank = tensor.shape[0], len(tensor.shape)
        it = multi_index_iterator(n_dimensions, rank)
        for x in it:
            weight_dict[it.multi_index] = base_weight * tensor[it.multi_index]
        return FactoredTensorWeight(weight_dict, rank, n_dimensions, base_weight, tensor)

@dataclass
class TensorWeightStack: # stack of TensorWeights to span desired space
    tw_list: list[TensorWeight]
    nonzero_indices: list[list[int]] # indices that need to be evaluated in product with this TensorWeight

    @staticmethod
    def make_stack(base_weight, n_dimensions, irrep): # choose TWS constructor based on irrep
        match irrep:
            case int():
                return full_stack(base_weight, n_dimensions, irrep)
            case FullRank():
                return full_stack(base_weight, n_dimensions, irrep.rank)
            case Antisymmetric(rank=2):
                return anti_rank2_stack(base_weight, n_dimensions)
            case SymmetricTraceFree(rank=2):
                return stf_rank2(base_weight, n_dimensions)
            case _:
                if irrep.rank == 0:
                    return scalar_stack(base_weight)
                if irrep.rank == 1:
                    return ones_vector_stack(base_weight, n_dimensions) # the general one should probably work fine too
                else:
                    warn(RuntimeWarning(f"Rank {irrep.rank} irreps may not be supported!")) 
                    return None        
    
    @staticmethod
    def scalar_stack(base_weight):
        tw = TensorWeight((), 0, None)
        nz_inds = [()]
        return TensorWeightStack([tw], nz_inds)
    
    @staticmethod
    def ones_vector_stack(base_weight, n_dimensions):
        tw_list = []
        for i in range(n_dimensions): 
            tensor = np.zeros(shape=(n_dimensions,))
            tensor[i] = 1
            tw_list.append(constant_tw(base_weight, tensor))
        nz_inds = [(i,) for i in range(n_dimensions)]
        return TensorWeightStack(tw_list)

    # see https://physics.stackexchange.com/questions/770127/some-intuitions-for-irreducible-representations-so3-in-classical-physics
    @staticmethod
    def stf_rank2_stack(base_weight, n_dimensions): # form all basis elements for rank 2 symmetric trace-free irrep
        tw_list = []
        nz_inds = []
        it = multi_index_iterator(n_dimensions, rank=2)
        for x in it:
            i, j = it.multi_index
            tensor = np.zeros(shape=(n_dimensions, n_dimensions))
            if i==j:
                nz_inds.append((i, i))
                if i==0: # no corresponding basis element
                    continue
                tensor[0, 0] = 1
                tensor[i, i] = -1
                tw_list.append(constant_tw(base_weight, tensor))
            elif i<j:
                tensor[i, j] = 1
                tensor[j, i] = 1
                tw_list.append(constant_tw(base_weight, tensor))
                nz_inds += [(i, j), (j, i)]
        return TensorWeightStack(tw_list)

    @staticmethod
    def anti_rank2_stack(base_weight, n_dimensions): # form all basis elements for rank 2 antisymmetric irrep
        tw_list = []
        nz_inds = []
        it = multi_index_iterator(n_dimensions, rank=2)
        for x in it:
            i, j = it.multi_index
            tensor = np.zeros(shape=(n_dimensions, n_dimensions))
            if i<j:
                tensor[i, j] = 1
                tensor[j, i] = -1
                tw_list.append(constant_tw(base_weight, tensor))
                nz_inds += [(i, j), (j, i)]
        return TensorWeightStack(tw_list)

    @staticmethod
    def full_stack(base_weight, n_dimensions, rank): # form all basis elements for rank n
        tw_list = []
        nz_inds = []
        it = multi_index_iterator(n_dimensions, rank)
        for x in it:
            tensor = np.zeros(shape=(n_dimensions,)*rank)
            tensor[it.multi_index] = 1
            tw_list.append(constant_tw(base_weight, tensor))
            nz_inds.append(it.multi_index)
        return TensorWeightStack(tw_list, nz_inds)

def lists_for_N(nloops, loop_max):
    if nloops == 0:
        yield []
        return
    for i in range(loop_max + 1):
        for li in lists_for_N(nloops - 1, loop_max):
            yield [i] + li
            

# this class might be absorbed into SRDataset
class LibraryData(object):  # structures information associated with a given rank (or irrep in general) library
    def __init__(self, terms, rank):  # , parent
        # self.parent = parent
        self.terms = terms
        self.rank = rank
        self.Q = None  # Q matrix
        self.col_weights = None
        self.row_weights = None
        

@dataclass(kw_only=True)
class AbstractDataset(object): # template for structure of all data associated with a given sparse regression dataset
    world_size: List[float] # linear dimensions of dataset in physical units (spatial + time)
    data_dict: Dict[Observable, np.ndarray[float]] # observable -> array of values (e.g. discrete - particle, spatial index, time)
    observables: List[Observable]  # list of observables
    # storage of computed quantities: (prim, domains) [not dims] -> array
    field_dict: dict[tuple[Any], np.ndarray[float]] = None 
    
    dxs: List[float] = None # grid spacings
    weight_dxs: List[float] = None
    scalar_weights: List[Weight] = None
    tensor_weight_stacks: Dict[Tuple[Union[int, Irrep, Weight]], TensorWeightStack] # (irrep, weight) -> stack
    # size of domain in grid units (NOT SUBGRID UNITS, AS ACTUALLY USED IN DISCRETE COMPUTATION)
    domain_size: List[float] = None 
    domains: List[IntegrationDomain] = None
    libs: Dict[Union[int, Irrep], LibraryData] = None # irrep label (e.g. 0, 1, "2s" irrep) -> LibraryData object
    irreps: List[Union[int, str]] = (0, 1) # set of irreducible representations to generate libraries for = libs.keys()

    scale_dict: Dict[str, float] = None # dict of characteristic scales of observables -> (mean, std)
    xscale: float = 1  # length scale for correct computation of char scales (default 1)
    tscale: float = 1  # time scale

    metric: Metric = None # we support only constant coeff metrics for now
    metric_is_identity: bool = True

    def __post_init__(self):
        self.n_dimensions = len(self.world_size) # number of dimensions (spatial + temporal)
        self.field_dict = dict()
        if self.metric is None: 
            self.metric = Metric(self.n_dimensions)
        else:
            self.metric_is_identity = False
    
    def make_libraries(self, max_complexity=4, max_observables=3, max_rho=999): # populate libs
        pass

    def make_domains(self, ndomains, domain_size, pad=0): # set domain_size/populate domains
        pass

    def make_weights(self, m, qmax): # populate weights/set weight_dxs
        self.weights = []
        self.weight_dxs = [(width - 1) / 2 * dx for width, dx in zip(self.domain_size, self.dxs)]
        for q in lists_for_N(self.n_dimensions, qmax):
            weight = Weight([m] * self.n_dimensions, q, [0] * self.n_dimensions, dxs=self.weight_dxs))
            self.weights.append(weight)
            for irrep in self.irreps:
                self.tensor_weight_stacks[irrep, weight] = TensorWeightStack.make_stack(weight, self.n_dimensions, irrep):

    # generate indexed term/weight pairs corresponding to concrete index of unindexed term
    def get_tw_pairs(self, term, base_weight, indices, debug=False): 
        for (space_orders, obs_dims) in self.get_dims(term, self.n_dimensions-1, indices):
            # first, make labeling canonical within each CGP
            if space_orders is None and obs_dims is None:
                space_orders = [[0] * self.n_dimensions for i in term.obs_list]
                canon_obs_dims = [[None] * i.cgp.rank if hasattr(i, 'cgp') else i.observable.rank for i in term.obs_list] 
            else:
                canon_obs_dims = []
                for sub_list, prim in zip(obs_dims, term.obs_list):
                    canon_obs_dims.append(prim.index_canon(sub_list))
            # integrate by parts
            indexed_term = IndexedTerm(term, space_orders, canon_obs_dims)
            if debug:
                print("ORIGINAL TERM:")
                print(indexed_term, [o.dimorders for o in indexed_term.obs_list])
            if by_parts:
                # integration by parts
                for mod_term, mod_weight in int_by_parts(indexed_term, base_weight):
                    if debug:
                        print("INTEGRATED BY PARTS:")
                        print(mod_term, [o.dimorders for o in mod_term.obs_list],
                              mod_weight)
                    yield mod_term, mod_weight
            else:
                yield indexed_term, base_weight
    
    def eval_term(self, term, domain, debug=False): # evaluate an IndexedTerm on domain (weight is removed from here)
        # term: IndexedTerm
        # weight
        # domain: IntegrationDomain corresponding to where the term is evaluated
        # return the evaluated term on the domain grid
        product = np.ones(shape=domain.shape)
        if isinstance(term, ConstantTerm): # short-circuit
            return product
        if debug:
            print(f"IndexedTerm {term}")
        for prim in term.obs_list:
            if debug:
                print(f"IndexedPrimitive {prim}")
            if (prim, domain) in self.field_dict.keys():  # field is "cached"
                data_slice = self.field_dict[prim, domain]
            else:
                data_slice = self.eval_prim(prim, domain)
                self.field_dict[prim, domain] = data_slice
            if sum(prim.dimorders) != 0:
                product *= diff(data_slice, prim.dimorders, self.dxs)
            else:
                product *= data_slice
            # print(product[0, 0, 0])
        return product

    # evaluate IndexedPrimitive on a domain - DIFFERENT IMPLEMENTATIONS for continuous and discrete!
    def eval_prim(self, prim, domain, *args): 
        pass

    # def trace(self, term_values, weight_values): # evaluate inner product of term evaluation and tensor weight using metric
    #     if self.metric_is_identity:
    #         return np.einsum('ij..., ij...->...', term_values, weight_values, optimize=True)
    #     else:
    #         return np.einsum('ij..., jk, ik...->...', term_values, self.metric, weight_values, optimize=True)

    def shortcut_trace(self, product_values, tensor): # evaluate inner product with factorizable TensorWeight
        if self.metric_is_identity:
            return np.einsum('ij..., ij->...', product_values, tensor, optimize=True)
        else:
            return np.einsum('ij..., jk, ik->...', product_values, self.metric, tensor, optimize=True)
    
    # def tuple_iterator(self, irrep): # make iterator over tuples (term, tensor_weight, domain) for a given irrep    
    #     for domain in domains:
    #         for term in self.libs[irrep].terms:
    #             for weight in self.weights:
    #                 for tensor_weight in self.tensor_weight_stacks[irrep, weight]: 
    #                     yield (term, tensor_weight, domain)

    def make_Q(self, irrep, by_parts=True, debug=False):
        #for tensor(term, tensor_weight, domain) in self.tuple_iterator(irrep):
        cols_list = []
        n_spatial_dims = self.n_dimensions-1
        if isinstance(irrep, SymmetryRep):
            rank = irrep.rank
        else:
            rank = irrep
            for term in self.libs[irrep].terms:
                column = []
                pv_dict = dict()
                for weight in self.weights:
                    for tensor_weight in self.tensor_weight_stacks[irrep, weight]:    
                        if self.metric_is_identity:
                            evaled_term_indices = tensor_weight.nonzero_indices
                        else: # there are better optimizations based on nonzero entries of metric but we will not include them
                            evaled_term_indices = lists_for_N(tensor_weight.rank, n_spatial_dims-1)
                        #arr = self.make_tw_arr(term, weight, domain, by_parts, debug)
                        base_weight = tensor_weight.base_weight  # only allow FactorizedTensorWeight for simplicity

                        for eti in evaled_term_indices:
                            # note that we have once again carefully taken integration by parts outside of the domain loop
                            for (indexed_term, new_weight) in self.get_tw_pairs(term, base_weight, eti, by_parts, debug):
                                for domain in domains: 
                                    # evaluate IndexedTerms & weights for each evaled_term_index
                                    # indices of arrays are (same indices as tensor of rank, same indices as domain)
                                    pv_dict[tensor_weight, domain] = np.zeros(shape=[n_spatial_dims]*rank+list(domain.shape))
                                    # (not term_value_array)
                                    #weight_value_array = np.zeros(shape=term_value_array.shape)
                                    pv_dict[tensor_weight, domain][*eti, ...] += 
                                        self.eval_term(self, indexed_term, domain, debug) * new_weight.get_weight_array(domain.shape)

                        # if tensor_weight.base_weight is None:
                        #    for inds in tensor_weight.nonzero_indices:
                        #        weight_value_array[*inds, ...] = tensor_weight[inds].get_weight_array(domain.shape)
                        #    arr = self.trace(term_value_array, weight_value_array)
                        # else:
                        #base_weight_arr = tensor_weight.base_weight.get_weight_array(domain.shape)

                        # fair question: should we integrate and then multiply by the tensor or multiply and then integrate?
                        for domain in domains:
                            arr = self.shortcut_trace(pv_dict[tensor_weight, domain], tensor_weight.tensor)
                            column.append(int_arr(arr, self.dxs)) # integrate the product array
                cols_list.append(column)
        return np.ndarray(cols_list).transpose # convert to numpy array
        
    def make_library_matrices(self, by_parts=True, debug=False): # compute LibraryData Q matrices
        for irrep in irreps:
            if debug:
                print(f"***RANK {irrep} LIBRARY***")
            self.libs[irrep].Q = make_Q(self, irrep, by_parts, debug)
        self.find_scales()
        for irrep in irreps:
            self.libs[irrep].col_weights = [self.get_char_size(term) for term in self.libs[irrep].terms]
        #self.find_row_weights()

    def find_scales(self, names=None): # find mean/std deviation of fields in data_dict that are in names
        pass

    def set_LT_scale(self, L, T): # compute correlation length/time
        pass

    def get_char_size(self, term): # get characteristic size of LibraryTerm 
        pass

def find_term(term_list, string):  # find first index of term in list matching string
    return [str(elt) for elt in term_list].index(string)

def get_term(term_list, string): # return first term in list matching string
    return next((elt for elt in term_list if str(elt) == string), None)

def get_slice(arr, domain):
    arr_slice = arr
    for (slice_dim, min_c, max_c) in zip(range(arr.ndim), domain.min_corner, domain.max_corner):
        idx = [slice(None)] * arr.ndim
        idx[slice_dim] = slice(min_c, max_c + 1)
        arr_slice = arr_slice[tuple(idx)]
    return arr_slice

def int_arr(arr, dxs=None):  # integrate the output of eval_term with respect to weight function
    if dxs is None:
        dxs = [1] * len(arr.shape)
    dx = dxs[0]
    integral = np.trapz(arr, axis=0)
    if len(dxs) == 1:
        return integral
    else:
        return int_arr(integral, dxs[1:])

def int_by_parts(term, weight, dim=0):
    if dim >= term.ndims:
        yield term, weight
    else:
        failed = False
        for te, we, fail in int_by_parts_dim(term, weight, dim):  # try to integrate by parts
            failed = (failed or fail)
            if failed:  # can't continue, so go to next dimension
                dim += 1
            yield from int_by_parts(te, we, dim)  # repeat process (possibly on next dimension)


# for integration by parts, check terms that look like x', x*x', and x*x*x' (vs all other terms have derivative orders
# smaller by at least 2) this misses out on opportunities to integrate by parts using a different basis, but this seems
# too difficult to automate; at that point it's probably easier to just write the term out manually.
def int_by_parts_dim(term, weight, dim):
    # find best term to base integration off of
    # best_prim, next_best, third_best = None, None, None
    # best_i, next_i, third_i = None, None, None
    best_prim, next_prim = None, None
    num_next = 0
    for (i, prim) in enumerate(term.obs_list):
        if prim.nderivs == term.nderivs:
            if best_prim is None:
                best_i, best_prim = i, prim
                if prim.dimorders[dim] == 0:  # can't integrate by parts along this dimension
                    yield term, weight, True
                    return
            else:  # multiple candidate terms -> integrating by parts will not help
                yield term, weight, True
                return
        elif prim.nderivs == term.nderivs - 1:
            if next_prim is None:
                num_next, next_prim = 1, prim
            elif prim == next_prim:
                num_next += 1
            else:  # not all one-lower terms are successors of best_prim
                yield term, weight, True
    # check viability by cases
    newords = copy.deepcopy(best_prim.dimorders)
    newords[dim] -= 1
    new_weight = weight.increment(dim)
    new_prim = IndexedPrimitive(best_prim, newords=newords)
    # print(term, best_prim)
    rest = term.drop(best_prim)
    if next_prim is None:  # then all other terms have derivatives up to order n-2, so we are in x' case
        for summand in rest.diff(dim):
            yield new_prim * summand, -weight, False
        yield new_prim * rest, -new_weight, False
        return
    else:
        # print(rest, next_prim)
        if next_prim.succeeds(best_prim, dim):  # check if next_best goes with best
            rest = rest.drop_all(next_prim)
            num_dupes = 1 + num_next
            for summand in rest.diff(dim):
                yield reduce(mul, [next_prim] * num_dupes) * summand, -1 / num_dupes * weight, False
            yield reduce(mul, [next_prim] * num_dupes) * rest, -1 / num_dupes * new_weight, False
            return
        else:
            yield term, weight, True
            return

def diff(data, dorders, dxs=None, acc=6):
    # for spatial directions can use finite differences or spectral differentiation. For time, only the former.
    # in any case, it's probably best to pre-compute the derivatives on the whole domains (at least up to order 2).
    # with integration by parts, there shouldn't be higher derivatives.
    if dxs is None:
        dxs = [1] * len(dorders)
    diff_list = []
    for i, dx, order in zip(range(len(dxs)), dxs, dorders):
        if order > 0:
            diff_list.append((i, dx, order))
    diff_operator = FinDiff(*diff_list, acc=acc)
    return diff_operator(data)