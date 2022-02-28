from functools import reduce
from operator import add
from itertools import permutations
from numpy import inf
import numpy as np
import copy

from dataclasses import dataclass, field
from typing import List

@dataclass(order=True)
class CompPair(object):
    torder: int
    xorder: int
    complexity: int = field(init=False)
        
    def __post_init__(self):
        self.complexity = self.torder + self.xorder
    def __repr__(self):
        return f'({self.torder}, {self.xorder})'

@dataclass
class CompList(object):
    in_list: List[int]
    
    def __ge__(self, other):
        for x, y in zip(self.in_list, other.in_list):
            if x>y:
                return True
            elif x<y:
                return False
        return True
    
    def special_bigger(self, other):
        if 0 in self.in_list and 0 not in other.in_list:
            return True
        elif 0 in other.in_list and 0 not in self.in_list:
            return False
        else:
            return self>=other

class DerivativeOrder(CompPair):
    def dt(self):
        return DerivativeOrder(self.torder+1, self.xorder)
    
    def dx(self):
        return DerivativeOrder(self.torder, self.xorder+1)

@dataclass
class Observable(object):
    string: str
    rank: int
    
    def __repr__(self):
        return self.string
    
    # For sorting: convention is in ascending order of name
    
    def __lt__ (self, other):
        if not isinstance(other, Observable):
            raise ValueError("Second argument is not an observable.") 
        return self.string<other.string

    def __gt__ (self, other):
        if not isinstance(other, Observable):
            raise ValueError("Second argument is not an observable.") 
        return other.__lt__(self)

    def __eq__ (self, other):
        if not isinstance(other, Observable):
            raise ValueError("Second argument is not an observable.") 
        return self.string==other.string

    def __ne__ (self, other):
        return not self.__eq__(other)

@dataclass
class LibraryPrimitive(object):
    dorder: DerivativeOrder
    observable: Observable
    rank: int = field(init=False)
    complexity: int = field(init=False)
    
    def __post_init__(self):
        self.rank = self.dorder.xorder + self.observable.rank
        self.complexity = self.dorder.complexity + 1
        
    def __repr__(self):
        torder = self.dorder.torder
        xorder = self.dorder.xorder
        if torder==0:
            tstring = ""
        elif torder==1:
            tstring = "dt "
        else:
            tstring = f"dt^{torder} "
        if xorder==0:
            xstring = ""
        elif xorder==1:
            xstring = "dx "
        else:
            xstring = f"dx^{xorder} "
        return f'{tstring}{xstring}{self.observable}'
    
    # For sorting: convention is (1) in ascending order of name/observable, (2) in DESCENDING order of dorder
    
    def __lt__ (self, other):
        if not isinstance(other, LibraryPrimitive):
            raise ValueError("Second argument is not a LibraryPrimitive.") 
        if self.observable == other.observable:
            return self.dorder > other.dorder
        else:
            return self.observable < other.observable

    def __gt__ (self, other):
        if not isinstance(other, LibraryPrimitive):
            raise ValueError("Second argument is not a LibraryPrimitive.") 
        return other.__lt__(self)

    def __eq__ (self, other):
        if not isinstance(other, LibraryPrimitive):
            raise ValueError("Second argument is not a LibraryPrimitive.") 
        return self.observable==other.observable and self.dorder==other.dorder

    def __ne__ (self, other):
        return not self.__eq__(other)
    
    def dt(self):
        return LibraryPrimitive(self.dorder.dt(), self.observable)
    
    def dx(self):
        return LibraryPrimitive(self.dorder.dx(), self.observable)
    
class IndexedPrimitive(LibraryPrimitive):
    dim_to_let = {0: 'x', 1: 'y', 2: 'z'}
    
    def __init__(self, prim, space_orders=None, obs_dim=None, newords=None):
        self.dorder = prim.dorder
        self.observable = prim.observable
        self.rank = prim.rank
        self.complexity = prim.complexity
        if newords is None: # normal constructor
            self.dimorders = space_orders+[self.dorder.torder]
            self.obs_dim = obs_dim
        else: # modifying constructor
            self.dimorders = newords
            self.obs_dim = prim.obs_dim
        self.ndims = len(self.dimorders)
        self.nderivs = sum(self.dimorders)
        
    def __repr__(self):
        torder = self.dimorders[-1]
        xstring = ""
        for i in range(len(self.dimorders)-1):
            let = self.dim_to_let[i]
            xorder = self.dimorders[i]
            if xorder==0:
                xstring += ""
            elif xorder==1:
                xstring += f"d{let} "
            else:
                xstring += f"d{let}^{xorder} "
        if torder==0:
            tstring = ""
        elif torder==1:
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
        return (self.dimorders==other.dimorders and self.observable==other.observable \
                and self.obs_dim==other.obs_dim)
    
    def __mul__(self, other):
        if isinstance(other, IndexedTerm):
            return IndexedTerm(observable_list=[self]+other.observable_list)
        else:
            return IndexedTerm(observable_list=[self, other])
    
    def succeeds(self, other, dim):
        copyorders = self.dimorders.copy()
        copyorders[dim] += 1
        return copyorders==other.dimorders and self.observable==other.observable and self.obs_dim==other.obs_dim
    
    def diff(self, dim):
        newords = self.dimorders.copy()
        newords[dim] += 1
        return IndexedPrimitive(self, newords=newords)
    
class LibraryTensor(object): # unindexed version of LibraryTerm
    def __init__(self, observables):
        if isinstance(observables, LibraryPrimitive):  # constructor for library terms consisting of a primitive with some derivatives
            self.observable_list = [observables]
        else:  # constructor for library terms consisting of a product
            self.observable_list = observables
        self.rank = sum([obs.rank for obs in self.observable_list])
        self.complexity = sum([obs.complexity for obs in self.observable_list])
        
    def __mul__(self, other):
        if isinstance(other, LibraryTensor):
            return LibraryTensor(self.observable_list + other.observable_list)
        elif other==1:
            return self
        else:
            raise ValueError(f"Cannot multiply {type(self)}, {type(other)}")
    
    def __rmul__(self, other):
        return __mul__(self, other)
    
    def __repr__(self):
        repstr = [str(obs)+' * ' for obs in self.observable_list]
        return reduce(add, repstr)[:-3]

def labels_to_index_list(labels, n): # n = number of observables
    index_list = [list() for i in range(2*n)]
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
    
def flatten(t):
    return [item for sublist in t for item in sublist]      

num_to_let_dict = {0: 'i', 1: 'j', 2: 'k', 3: 'l', 4: 'm', 5: 'n', 6: 'p'}
let_to_num_dict = {v: k for k, v in num_to_let_dict.items()} # inverted dict
    
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
    canon_dict = dict() # used to store ambiguous canonicalizations (which shouldn't exist for less than 6 indices)
    
    def __init__(self, libtensor, labels=None, index_list=None):
        self.observable_list = libtensor.observable_list
        self.libtensor = libtensor
        self.rank = (libtensor.rank % 2)
        self.complexity = libtensor.complexity
        if labels is not None: # from labels constructor
            self.labels = labels # dictionary: key = index #, value(s) = location of index among 2n bins
            self.index_list = labels_to_index_list(labels, len(self.observable_list))
        else: # from index_list constructor
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
            return self.observable_list==other.observable_list and self.index_list==other.index_list
        else:
            return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        return str(self)<str(other)
    
    def __gt__(self, other):
        return str(self)>str(other)
    
    def __repr__(self):
        repstr = [label_repr(obs, ind1, ind2)+' * ' for (obs, ind1, ind2) in zip(self.observable_list, num_to_let(self.der_index_list), num_to_let(self.obs_index_list))]
        return reduce(add, repstr)[:-3]
    
    def __hash__(self): # it's nice to be able to use LibraryTerms in sets or dicts
        return hash(self.__repr__())
    
    def __mul__(self, other):
        if isinstance(other, LibraryTerm):
            if self.rank < other.rank:
                return other.__mul__(self)
            if len(self.labels.keys()) > 0:
                shift = max(self.labels.keys())
            else:
                shift = 0
            if other.rank==1:
                a, b = self.increment_indices(1), other.increment_indices(shift+1)
            else:
                a, b = self, other.increment_indices(shift)
            return LibraryTerm(LibraryTensor(a.observable_list + b.observable_list),
                               index_list = a.index_list + b.index_list).canonicalize()
        elif str(other)=="1":
            return self
        elif isinstance(other, Equation):
            return other.__mul__(self)
        else:
            raise ValueError(f"Cannot multiply {type(self)}, {type(other)}")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def structure_canonicalize(self):
        indexed_zip = zip(self.observable_list, self.der_index_list, self.obs_index_list)
        sorted_zip = sorted(indexed_zip, key=lambda x:x[0])
        sorted_obs = [e[0] for e in sorted_zip]
        sorted_ind1 = [e[1] for e in sorted_zip]
        sorted_ind2 = [e[2] for e in sorted_zip]
        sorted_ind = flatten(list(zip(sorted_ind1, sorted_ind2)))
        sorted_libtens = LibraryTensor(sorted_obs)
        return LibraryTerm(sorted_libtens, index_list=sorted_ind)
    
    def index_canonicalize(self):
        #inc = 0
        #if len(self.labels[0])==2: # if multiple i's, need to increment all indices
        #    inc = 1
        subs_dict = canonicalize_indices(flatten(self.index_list))
        new_index_list = [[subs_dict[i] for i in li] for li in self.index_list]
        return LibraryTerm(self.libtensor, index_list=new_index_list)
    
    def reorder(self, template):
        indexed_zip = zip(self.observable_list, self.der_index_list, self.obs_index_list, template)
        sorted_zip = sorted(indexed_zip, key=lambda x:x[3])
        sorted_obs = [e[0] for e in sorted_zip]
        sorted_ind1 = [e[1] for e in sorted_zip]
        sorted_ind2 = [e[2] for e in sorted_zip]
        sorted_ind = flatten(list(zip(sorted_ind1, sorted_ind2)))
        sorted_libtens = LibraryTensor(sorted_obs)
        return LibraryTerm(sorted_libtens, index_list=sorted_ind)
    
    def canonicalize(self): # return canonical representation and set is_canonical flag (used to determine if valid)
        str_canon = self.structure_canonicalize()
        if str_canon in self.canon_dict:
            canon = self.canon_dict[str_canon]
            self.is_canonical = (self==canon)
            return canon
        alternative_canons = []
        for template in get_isomorphic_terms(str_canon.observable_list):
            term = str_canon.reorder(template)
            canon_term = term.index_canonicalize()
            alternative_canons.append(canon_term)
        canon = min(alternative_canons, key=str)
        for alt_canon in alternative_canons:
            self.canon_dict[alt_canon] = canon
        self.is_canonical = (self==canon)
        return canon
    
    def increment_indices(self, inc):
        index_list = [[index+inc for index in li] for li in self.index_list]
        return LibraryTerm(LibraryTensor(self.observable_list), index_list=index_list)
    
    def dt(self):
        terms = []
        for i, obs in enumerate(self.observable_list):
            new_obs = obs.dt()
            # note: no need to recanonicalize terms after a dt
            lt = LibraryTerm(LibraryTensor(self.observable_list[:i]+[new_obs]+self.observable_list[i+1:]), 
                             index_list=self.index_list)
            terms.append(lt)
        ts = TermSum(terms)
        return ts.canonicalize()
    
    def dx(self):
        terms = []
        for i, obs in enumerate(self.observable_list):
            new_obs = obs.dx()
            new_index_list = copy.deepcopy(self.index_list)
            new_index_list[2*i].insert(0, 0)
            lt = LibraryTerm(LibraryTensor(self.observable_list[:i]+[new_obs]+self.observable_list[i+1:]),
                             index_list=new_index_list)
            if lt.rank == 0:
                lt = lt.increment_indices(1)
            lt = lt.canonicalize() # structure changes after derivative so we must recanonicalize
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
    while reps<len(obs_list) and prev == obs_list[reps]:
        reps += 1
    for new_list in get_isomorphic_terms(obs_list[reps:], start_order[reps:]):
        for perm in permutations(start_order[:reps]):
            yield list(perm)+new_list
            
class IndexedTerm(object): # LibraryTerm with i's mapped to x/y/z
    def __init__(self, libterm=None, space_orders=None, obs_dims=None, observable_list=None):
        if observable_list is None: # normal "from scratch" constructor
            self.rank = libterm.rank
            self.complexity = libterm.complexity
            nterms = len(libterm.observable_list)
            self.observable_list = libterm.observable_list.copy()
            for i, obs, sp_ord, obs_dim in zip(range(nterms), libterm.observable_list, space_orders, obs_dims):
                self.observable_list[i] = IndexedPrimitive(obs, sp_ord, obs_dim)
            self.ndims = len(space_orders[0])+1
            self.nderivs = np.max([p.nderivs for p in self.observable_list])
        else: # direct constructor from observable list
            if len(observable_list)>0: # if term is not simply equal to 1
                self.rank = observable_list[0].rank
                self.ndims = observable_list[0].ndims
                self.observable_list = observable_list
                self.complexity = sum([obs.complexity for obs in observable_list])
                self.nderivs = np.max([p.nderivs for p in self.observable_list])
            else:
                self.observable_list = []
                self.ndims = 0
                self.nderivs = 0
                self.complexity = 0
            
    def __repr__(self):
        repstr = [str(obs)+' * ' for obs in self.observable_list]
        return reduce(add, repstr)[:-3]
    
    def drop(self, obs):
        obs_list_copy = self.observable_list.copy()
        if len(obs_list_copy)>1:
            obs_list_copy.remove(obs)
        else:
            obs_list_copy = []
        return IndexedTerm(observable_list=obs_list_copy)
    
    def diff(self, dim):
        for i, obs in enumerate(self.observable_list):
            yield obs.diff(dim)*self.drop(obs)
            
# Note: must be handled separately in derivatives
class ConstantTerm(IndexedTerm):
    def __init__(self):
        self.observable_list = []
        self.rank = 0
        self.complexity = 1
                
    def __repr__(self):
        return "1"
    
def label_repr(prim, ind1, ind2):
    torder = prim.dorder.torder
    xorder = prim.dorder.xorder
    obs = prim.observable
    if torder==0:
        tstring = ""
    elif torder==1:
        tstring = "dt "
    else:
        tstring = f"dt^{torder} "
    if xorder==0:
        xstring = ""
    else:
        ind1 = compress(ind1)
        xlist = [f"d{letter} " for letter in ind1]
        xstring = reduce(add, xlist)
    if obs.rank == 1:
        obstring = obs.string+"_"+ind2[0]
    else:
        obstring = obs.string
    return tstring+xstring+obstring

def compress(labels):
    copy = []
    skip = False
    for i in range(len(labels)):
        if i<len(labels)-1 and labels[i]==labels[i+1]:
            copy.append(labels[i]+'^2')
            skip = True
        elif not skip:
            copy.append(labels[i])
        else:
            skip = False
    return copy
                           
# make a dictionary of how paired indices are placed
def place_pairs(*rank_array, min_ind2=0, curr_ind=1, start=0, answer_dict=dict()):
    while rank_array[start]<=0:
        start += 1
        min_ind2 = 0
        if start>=len(rank_array):
            yield answer_dict
            return
    ind1 = start
    for ind2 in range(min_ind2, len(rank_array)):
        if (ind1==ind2 and rank_array[ind1]==1) or rank_array[ind2]==0:
            continue
        min_ind2 = ind2
        dict1 = answer_dict.copy()
        dict1.update({curr_ind: (ind1, ind2)})
        copy_array = np.array(rank_array)
        copy_array[ind1] -= 1
        copy_array[ind2] -= 1
        yield from place_pairs(*copy_array, min_ind2=min_ind2, curr_ind=curr_ind+1, start=start, answer_dict=dict1)
            
def place_indices(*rank_array):
    # only paired indices allowed
    if sum(rank_array) % 2 == 0:
        yield from place_pairs(*rank_array)
    # one single index
    else:
        for single_ind in range(len(rank_array)):
            if rank_array[single_ind]>0:
                copy_array = np.array(rank_array)
                copy_array[single_ind] -= 1
                yield from place_pairs(*copy_array, answer_dict={0: [single_ind]})

def list_labels(tensor):
    rank_array = []
    for term in tensor.observable_list:
        rank_array.append(term.dorder.xorder)
        rank_array.append(term.observable.rank)
    return [output_dict for output_dict in place_indices(*rank_array) if test_valid_label(output_dict, tensor.observable_list)]

# check if index labeling is invalid (i.e. not in non-decreasing order among identical terms)
# this excludes more incorrect options early than is_canonical
# the lexicographic ordering rule fails at N=6 but this is accounted for by the canonicalization
def test_valid_label(output_dict, obs_list):
    if len(output_dict.keys())<2: # not enough indices for something to be invalid
        return True
    # this can be implemented more efficiently, but the cost is negligible for reasonably small N
    bins = [] # bin observations according to equality
    for obs in obs_list:
        found_match = False
        for bi in bins:
            if bi is not None and obs==bi[0]:
                bi.append(obs)
                found_match = True
        if not found_match:
            bins.append([obs])
    if len(bins)==len(obs_list):
        return True # no repeated values
    # else need to check more carefully
    n = len(obs_list)
    index_list = labels_to_index_list(output_dict, n)
    for i in range(n):
        for j in range(i+1, n):
            if obs_list[i] == obs_list[j]:
                clist1 = CompList(index_list[2*i]+index_list[2*i+1])
                clist2 = CompList(index_list[2*j]+index_list[2*j+1])
                if not clist2.special_bigger(clist1): # if (lexicographic) order decreases OR i appears late
                    return False
    return True

def raw_library_tensors(observables, obs_orders, nt, nx, max_order=DerivativeOrder(inf, inf), zeroidx=0):
    #print(obs_orders, nt, nx, max_order)
    while obs_orders[zeroidx]==0:
        zeroidx += 1
        if zeroidx==len(observables):
            yield 1
            return
    if sum(obs_orders)==1:
        i = obs_orders.index(1)
        do = DerivativeOrder(nt, nx)
        if max_order>=do:
            prim = LibraryPrimitive(do, observables[i])
            yield LibraryTensor(prim)
        return
    for i in range(nt+1):
        for j in range(nx+1):
            do = DerivativeOrder(i, j)
            if max_order>=do:
                prim = LibraryPrimitive(do, observables[zeroidx]) 
                term1 = LibraryTensor(prim)
                new_orders = list(obs_orders)
                new_orders[zeroidx] -= 1
                if obs_orders[zeroidx]==1: # reset max_order since we are going to next terms
                    do = DerivativeOrder(inf, inf)
                for term2 in raw_library_tensors(observables, new_orders, nt-i, nx-j, max_order=do):
                    yield term1*term2

rho = Observable('rho', 0)
v = Observable('v', 1)
def generate_terms_to(order, observables=[rho, v], max_observables=999):
    observables = sorted(observables) # make sure ordering is consistent with canonicalization rules
    libterms = list()
    libterms.append(ConstantTerm())
    N = order # max number of "blocks" to include
    K = len(observables)
    part = partition(N, K+2) # K observables + 2 derivative dimensions
    # not a valid term if no observables or max exceeded
    for part in partition(N, K+2):
        #print(part)
        if sum(part[:K])>0 and sum(part[:K])<=max_observables:
            nt, nx = part[-2:]
            obs_orders = part[:-2]
            for tensor in raw_library_tensors(observables, obs_orders, nt, nx):
                for label in list_labels(tensor):
                    lt = LibraryTerm(tensor, label)
                    canon = lt.canonicalize()
                    if lt.is_canonical:
                        libterms.append(lt)
    return libterms

def partition(n,k):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    if k < 1:
        return
    if k == 1:
        for i in range(n+1):
            yield (i,)
        return
    for i in range(n+1):
        for result in partition(n-i,k-1):
            yield (i,)+result
            
class Equation(object): # can represent equation (expression = 0) OR expression
    def __init__(self, term_list, coeffs): # terms are LibraryTerms, coeffs are real numbers
        content = zip(term_list, coeffs)
        sorted_content = sorted(content, key=lambda x:x[0])
        # note that sorting guarantees canonicalization in equation term order
        self.term_list = [e[0] for e in sorted_content]
        self.coeffs = [e[1] for e in sorted_content]
        self.rank = term_list[0].rank
        self.complexity = sum([term.complexity for term in term_list]) # another choice is simply the number of terms
      
    def __add__(self, other):
        if isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise ValueError(f"Second argument {other}) is not an equation.")
            
    def __rmul__(self, other):
        if isinstance(other, LibraryTerm):
            return Equation([other*term for term in self.term_list], self.coeffs)
        else: # multiplication by number
            return Equation(self.term_list, [other*c for c in self.coeffs])
        
    def __mul__(self, other):
        return self.__rmul__(other)
    
    def __repr__(self):
        repstr = [str(coeff) + ' * ' + str(term)+' + ' for term, coeff in zip(self.term_list, self.coeffs)]
        return reduce(add, repstr)[:-3]
    
    def __str__(self):
        return self.__repr__()+" = 0"
    
    def __eq__(self, other):
        for term, ot in zip(self.term_list, other.term_list):
            if term != ot:
                return False
        for coeff, ot in zip(self.coeffs, other.coeffs):
            if term != ot:
                return False
        return True
    
    def dt(self):
        components = [coeff*term.dt() for term, coeff in zip(self.term_list, self.coeffs)
                      if not isinstance(term, ConstantTerm)]
        return reduce(add, components).canonicalize()
        
    def dx(self):
        components = [coeff*term.dx() for term, coeff in zip(self.term_list, self.coeffs)
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
        if len(self.term_list)==1:
            return self.term_list[0], None
        lhs = max(self.term_list, key=lambda t:t.complexity)
        lhs_ind = self.term_list.index(lhs)
        new_term_list = self.term_list[:lhs_ind]+self.term_list[lhs_ind+1:]
        new_coeffs = self.coeffs[:lhs_ind]+self.coeffs[lhs_ind+1:]
        new_coeffs = [-c/self.coeffs[lhs_ind] for c in new_coeffs]
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
    def __init__(self, term_list): # terms are LibraryTerms, coeffs are real numbers
        self.term_list = sorted(term_list)
        self.coeffs = [1]*len(term_list)
        self.rank = term_list[0].rank
        
    def __str__(self):
        repstr = [str(term)+' + ' for term in self.term_list]
        return reduce(add, repstr)[:-3]
    
    def __add__(self, other):
        if isinstance(other, TermSum):
            return TermSum(self.term_list + other.term_list, self.coeffs + other.coeffs)
        elif isinstance(other, Equation):
            return Equation(self.term_list + other.term_list, self.coeffs + other.coeffs)
        else:
            raise ValueError(f"Second argument {other}) is not an equation.")