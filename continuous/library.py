from functools import reduce
from operator import add
from itertools import permutations
from numpy import inf
import numpy as np

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
    pass

@dataclass
class Observable(object):
    string: str
    rank: int
    
    def __repr__(self):
        return self.string

    def __str__(self):
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
    simple: bool = True
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
    
    def __str__(self):
        return self.__repr__()
    
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
    
class IndexedPrimitive(LibraryPrimitive):
    dim_to_let = {0: 'x', 1: 'y', 2: 'z'}
    
    def __init__(self, prim, space_orders=None, obs_dim=None, newords=None): #parity = 1
        self.simple = True
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
        #self.parity = parity
        
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
        #if self.parity == -1:
        #    pstring = ""
        #else:
        #    pstring = "-"
        #return f'{pstring}{tstring}{xstring}{self.observable}{dimstring}'
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
        if isinstance(observables, LibraryPrimitive):  # constructor for library terms consisting of an observable with some derivatives
            self.simple = True
            self.observable_list = [observables]
        else:  # constructor for library terms consisting of a product
            self.simple = False
            self.observable_list = observables
        self.rank = sum([obs.rank for obs in self.observable_list])
        self.complexity = sum([obs.complexity for obs in self.observable_list])
        
    def __mul__(self, other):
        if isinstance(other, LibraryTensor):
            return LibraryTensor(self.observable_list + other.observable_list)
        elif other==1:
            return self
        else:
            raise ValueError(f"Cannot multiply {self}, {other}")
    
    def __rmul__(self, other):
        return __mul__(self, other)
    
    def __repr__(self):
        repstr = [str(obs)+' * ' for obs in self.observable_list]
        return reduce(add, repstr)[:-3]
    
    def __str__(self):
        return self.__repr__()

def labels_to_index_list(labels, n): # n = number of observables
    index_list = [list() for i in range(2*n)]
    for key in labels.keys():
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

class LibraryTerm(object):
    canon_dict = dict() # used to store ambiguous canonicalizations (which shouldn't exist for less than 6 indices)
    
    def __init__(self, libtensor, labels=None, index_list=None):
        self.observable_list = libtensor.observable_list
        self.libtensor = libtensor
        self.rank = (libtensor.rank % 2)
        self.complexity = libtensor.complexity
        if labels is not None: # from labels constructor
            self.labels = labels # dictionary: key = index #, value(s) = location of index among 2n bins
            #self.index_list = [list() for i in range(len(self.observable_list)*2)] # list: indices in each of 2n bins
            self.index_list = labels_to_index_list(labels, len(self.observable_list))
        else: # from index_list constructor
            self.index_list = index_list
            self.labels = index_list_to_labels(index_list)
        self.der_index_list = self.index_list[0::2]
        self.obs_index_list = self.index_list[1::2]
        self.is_canonical = None
        #for key in labels.keys():
        #    letter = self.num_to_let[key]
        #    for a in labels[key]:
        #        self.index_list[a].append(letter)
#         self.index_list = [list() for i in range(len(self.observable_list)*2)] 
#         if len(labels)>0:
#             num_indices = [(obs.dorder.xorder, obs.observable.rank) for obs in self.observable_list]
#             num_indices = flatten(num_indices)
#             #print(num_indices)
#             max_label = max(labels)
#             self.labels = dict()
#             for a in range(max_label+1):
#                 self.labels[a] = []
#             i = 0
#             j = 0
#             #print(self.labels)
#             for n in num_indices:
#                 self.index_list[j] = labels[i:i+n]
#                 for key in self.index_list[j]:
#                     self.labels[key].append(j)
#                 self.index_list[j] = [self.num_to_let[key] for key in self.index_list[j]]
#                 i += n
#                 j += 1
#         else:
#             self.labels = dict()
#         #print(self.index_list, self.labels)
                
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
    
    def __str__(self):
        return self.__repr__()
    
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
        #new_index_list = [list() for i in range(len(self.observable_list)*2)] 
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
        if str(str_canon) in self.canon_dict:
            canon = self.canon_dict[str(str_canon)]
            self.is_canonical = (self==canon)
            return canon
        alternative_canons = []
        for template in get_isomorphic_terms(str_canon.observable_list):
            term = str_canon.reorder(template)
            canon_term = term.index_canonicalize()
            alternative_canons.append(canon_term)
        canon = min(alternative_canons, key=str)
        for alt_canon in alternative_canons:
            self.canon_dict[str(alt_canon)] = canon
        self.is_canonical = (self==canon)
        return canon
    
def get_isomorphic_terms(obs_list, start_order=None):
    if start_order is None:
        start_order = list(range(len(obs_list)))
    if len(obs_list) == 0:
        yield []
        return
    reps = 1
    prev = obs_list[0]
    #if prev is None or prev.rank < 2:
    #    for new_list in get_isomorphic_terms(obs_list[1:], start_order[1:], obs_list[0])
    #        yield start_order[0]+new_list
    #else:
    while reps<len(obs_list) and prev == obs_list[reps]:
        reps += 1
    for new_list in get_isomorphic_terms(obs_list[reps:], start_order[reps:]):
        for perm in permutations(start_order[:reps]):
            yield list(perm)+new_list
            
class IndexedTerm(object): # LibraryTerm with i's mapped to x/y/z
    def __init__(self, libterm=None, space_orders=None, obs_dims=None, observable_list=None): #indterm=None, neworders=None,
        if observable_list is None: # normal "from scratch" constructor
            self.rank = libterm.rank
            self.complexity = libterm.complexity
            #self.obs_dims = obs_dims
            nterms = len(libterm.observable_list)
            self.observable_list = libterm.observable_list.copy()
            for i, obs, sp_ord, obs_dim in zip(range(nterms), libterm.observable_list, space_orders, obs_dims):
                self.observable_list[i] = IndexedPrimitive(obs, sp_ord, obs_dim)
            self.ndims = len(space_orders[0])+1
            self.nderivs = np.max([p.nderivs for p in self.observable_list])
        #elif indterm is not None: # integrate by parts constructor
        #    self.rank = indterm.rank
        #    #self.obs_dims = indterm.obs_dims
        #    self.observable_list = indterm.observable_list.copy()
        #    for prim, ords, obs_dim in zip(indterm.observable_list, neworders, obs_dims):
        #        self.observable_list[i] = IndexedPrimitive(obs, obs_dim=obs_dim, newords=newords)
        #    self.ndims = len(neworders[0])
        else: # direct constructor from observable list
            #print(observable_list)
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
    
    def __mul__(self, other):
        if isinstance(other, IndexedTerm):
            return IndexedTerm(observable_list=self.observable_list+other.observable_list)
        else:
            return IndexedTerm(observable_list=self.observable_list+[other])
    
    def drop(self, obs):
        #print(self.observable_list)
        obs_list_copy = self.observable_list.copy()
        if len(obs_list_copy)>1:
            obs_list_copy.remove(obs)
        else:
            obs_list_copy = []
        return IndexedTerm(observable_list=obs_list_copy)
    
    def diff(self, dim):
        for i, obs in enumerate(self.observable_list):
            yield obs.diff(dim)*self.drop(obs)
            
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
                        
def raw_library_tensors(observables, obs_orders, nt, nx, max_order=DerivativeOrder(inf, inf), zeroidx=0):
    #print(obs_orders, nt, nx, max_order)
    while obs_orders[zeroidx]==0:
        zeroidx += 1
        if zeroidx==len(observables):
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
            if max_order>=DerivativeOrder(i, j):
                do = DerivativeOrder(i, j) 
                prim = LibraryPrimitive(do, observables[zeroidx]) 
                term1 = LibraryTensor(prim)
                new_orders = list(obs_orders)
                new_orders[zeroidx] -= 1
                if obs_orders[zeroidx]==1: # reset max_order since we are going to next terms
                    do = DerivativeOrder(inf, inf)
                for term2 in raw_library_tensors(observables, new_orders, nt-i, nx-j, max_order=do):
                    yield term1*term2
                        
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

def list_labels(tensor):#, tensor_scaffold):
    rank_array = []
    for term in tensor.observable_list:
        rank_array.append(term.dorder.xorder)
        rank_array.append(term.observable.rank)
    return [output_dict for output_dict in place_indices(*rank_array) if test_valid_label(output_dict, tensor.observable_list)]
    #return [output_dict for output_dict in place_indices(*rank_array) 
    #        if valid_canon_labeling(output_dict, tensor_scaffold)]

# the lexicographic ordering rule is only tested up to N=5 and likely fails soon afterwards
# use the more complex & complete unique_terms.py if needed -> more integrated implementation should be done here
# check if index labeling is valid (i.e. in non-decreasing order among identical terms)
# this excludes more incorrect options early than is_canonical
def test_valid_label(output_dict, obs_list):
    #index_list = labels_to_index_list(output_dict, len(obs_list))
    #return is_canonical(flatten(index_list))
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
    index_list = [list() for i in range(len(obs_list)*2)]
    for key in output_dict.keys():
        for a in output_dict[key]:
            index_list[a].append(key)
    #index_dict = dict()
    #for i in range(len(obs_list)):
    #    index_dict[obs_list[i]] = (index_list[2*i], index_list[2*i+1])
    for i in range(len(obs_list)):
        for j in range(i+1, len(obs_list)):
            if obs_list[i] == obs_list[j]:
                clist1 = CompList(index_list[2*i]+index_list[2*i+1])
                clist2 = CompList(index_list[2*j]+index_list[2*j+1])
                if not clist2.special_bigger(clist1): # if (lexicographic) order decreases OR i appears late
                #if not clist1<=clist2 or 0 in clist2.in_list:
                    return False

    # this is only guaranteed to work if there is only 1 index per identical term
    #for key1 in output_dict.keys():
    #    for key2 in output_dict.keys():
    #        if key1<key2: # we only test each pair once
    #            for val1 in output_dict[key1]:
    #                for val2 in output_dict[key2]:
    #                    if obs_list[val1//2]==obs_list[val2//2] and val1>val2:
    #                        return False # violation: decreasing label

    # if we got this far, the labeling is valid
    return True

rho = Observable('rho', 0)
v = Observable('v', 1)
def generate_terms_to(order, observables=[rho, v], max_observables=999):
    observables = sorted(observables) # make sure ordering is consistent with canonicalization rules
    libterms = list()
    libterms.append(ConstantTerm())
    N = order # max number of "blocks" to include
    K = len(observables)
    part = partition(N, K+2) # K observables + 2 derivative dimensions
    #maxs = [max_observables]*K+[np.inf]*2
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
                        libterms.append(LibraryTerm(tensor, label))
#                 indexing_guide = []
#                 rank = tensor.rank
#                 if rank%2 == 1:
#                     indexing_guide.append((1, 1))
#                 indexing_guide.append((2, rank//2))
#                 #print(tensor)
#                 tensor_scaffold = get_scaffold(tensor.observable_list)
#                 #print(tensor_scaffold)
#                 for var in unique_indexings(tensor_scaffold, indexing_guide):
#                     #print(var)
#                     cvar = canonicalize(var, {})
#                     if str(var) == str(cvar): # properly canonicalized
#                         #print('ok')
#                         stv = str(var)
#                         if stv!="None":
#                             indices = list(map(int, str(var).split()))
#                             #print(indices)
#                             if rank%2 == 0: # need to shift i->j etc.
#                                 indices = [ind+1 for ind in indices]
#                             libterms.append(LibraryTerm(tensor, indices))
#                         else:
#                             #print("None")
#                             libterms.append(LibraryTerm(tensor, []))
#                     else:
#                         raise ValueError(f"Bad canonicalization {var}")
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
            
class TermSum(object):
    def __init__(self, term_list):
        self.term_list = term_list
        self.rank = term_list[0].rank
      
    def __add__(self, other):
        if isinstance(other, TermSum):
            return TermSum(self.term_list + other.term_list)
        else:
            return TermSum(self.term_list + [other])
    
    def __repr__(self):
        repstr = [str(term)+' + ' for term in self.term_list]
        return reduce(add, repstr)[:-3]

# def get_scaffold(obs_list):
#     if len(obs_list)>1:
#         i = 1
#         old_obs = obs_list[0]
#         obs = obs_list[1]
#         while old_obs==obs:
#             i += 1
#             if i==len(obs_list):
#                 break
#             obs = obs_list[i]
#         if i>1:
#             sc = get_scaffold([old_obs])
#             if sc is not None:
#                 rs = RepeatScaffold(sc, i)
#             else:
#                 rs = None
#         else:
#             rs = get_scaffold([old_obs])
#         list_rem = obs_list[i:]
#         if len(list_rem)>0:
#             remsc = get_scaffold(list_rem)
#         else:
#             remsc = None
#         if remsc is not None and rs is not None:
#             return ProductScaffold(rs, get_scaffold(list_rem))
#         elif rs is not None:
#             return rs
#         elif remsc is not None:
#             return get_scaffold(list_rem)
#         else:
#             return None
#     else:
#         obs = obs_list[0]
#         xorder = obs.dorder.xorder
#         rank = obs.observable.rank
#         if xorder>0:
#             if xorder>1:
#                 rs1 = RepeatScaffold(Index(), xorder)
#             else:
#                 rs1 = Index()
#         if rank>0:
#             if rank>1:
#                 rs2 = RepeatScaffold(Index(), rank)
#             else:
#                 rs2 = Index()
#         if xorder>0 and rank>0:
#             return ProductScaffold(rs1, rs2)
#         elif xorder>0 and rank==0:
#             return rs1
#         elif xorder==0 and rank>0:
#             return rs2
#         else:
#             return None

### Akash code ###
# from dataclasses import dataclass, field
# from typing import TypeVar, Union, Generator

# T = TypeVar('T')

# IndexingScaffold = Union['Index', 'RepeatScaffold', 'ProductScaffold']


# @dataclass(frozen=True)
# class Index:
#     value: int = -1
#     blocked_size: (int, int) = (0, 0)
#     capacity: int = field(init=False)

#     def __post_init__(self):
#         object.__setattr__(
#             self, 'capacity', 1 if (self.value < 0) else 0)

#     def __str__(self):
#         if self.value < 0 and self.blocked_size > (0, 0):
#             return "(!{})".format(self.blocked_size)
#         return str(self.value) if self.value >= 0 else "?"


# @dataclass(frozen=True)
# class RepeatScaffold:
#     template: IndexingScaffold
#     n: int
#     blocked_size: (int, int) = (0, 0)
#     capacity: int = field(init=False)

#     def __post_init__(self):
#         object.__setattr__(
#             self, 'capacity', self.n * self.template.capacity)

#     def __str__(self):
#         return "({})^{}".format(self.template, self.n)


# @dataclass(frozen=True)
# class ProductScaffold:
#     l: IndexingScaffold
#     r: IndexingScaffold
#     blocked_size: (int, int) = (0, 0)
#     capacity: int = field(init=False)

#     def __post_init__(self):
#         object.__setattr__(
#             self, 'capacity', self.l.capacity + self.r.capacity)

#     def __str__(self):
#         return str(self.l) + " " + str(self.r)


# def block_size(s: IndexingScaffold, size):
#     if isinstance(s, Index):
#         return Index(s.value, size)
#     elif isinstance(s, RepeatScaffold):
#         return RepeatScaffold(s.template, s.n, blocked_size=size)
#     elif isinstance(s, ProductScaffold):
#         return ProductScaffold(s.l, s.r, blocked_size=size)


# def depth(s: IndexingScaffold):
#     if isinstance(s, Index):
#         return 1
#     elif isinstance(s, RepeatScaffold):
#         return 1 + depth(s.template)
#     elif isinstance(s, ProductScaffold):
#         return 1 + max(depth(s.l), depth(s.r))


# def iter_variations_(
#     s: IndexingScaffold, cur_i: int, i_size: int,
#     ct_added: int, ct_to_add: int, first_instance: bool
# ) -> Generator[IndexingScaffold, None, None]:
#     # print("   "*(5-depth(s)), s, "<<", cur_i, "*", ct_to_add)
#     if ct_to_add == 0:
#         # Key idea: if I choose not to add an index at a
#         # location, then I should not be able to add an
#         # equivalent index later on. Hence I block the location.
#         yield block_size(s, (i_size, ct_added)) if first_instance else s
#     elif (i_size == s.blocked_size[0] 
#               and ct_added + ct_to_add > s.blocked_size[1]
#          ) or ct_to_add > s.capacity:
#         # if i_size == s.blocked_size[0] and ct_added > s.blocked_size[1]:
#         #     print(" "*5, "!!", s, s.blocked_size, i_size, ct_to_add)
#         # elif ct_to_add > s.capacity:
#         #     print(" "*5, "??", s, ct_to_add, s.capacity)
#         return
#     elif isinstance(s, Index):
#         yield Index(cur_i)
#     elif isinstance(s, RepeatScaffold):
#         yield from iter_variations_repeat_(
#             s, cur_i, i_size, ct_added, ct_to_add, first_instance, 
#             min(ct_to_add, s.template.capacity))
#     elif isinstance(s, ProductScaffold):
#         min_ct = max(0, ct_to_add - s.r.capacity) - 1
#         max_ct = min(ct_to_add, s.l.capacity)
#         # print(s, min_ct, max_ct)
#         for ct in range(max_ct, min_ct, -1):
#             for lhs in iter_variations_(
#                     s.l, cur_i, i_size, ct_added, ct, first_instance):
#                 if first_instance:
#                     lhs = block_size(lhs, (i_size, ct_added + ct))
#                 for rhs in iter_variations_(
#                         s.r, cur_i, i_size, ct_added + ct, ct_to_add - ct,
#                         first_instance and ct == 0):
#                     yield ProductScaffold(lhs, rhs)


# def iter_variations_repeat_(
#     s: RepeatScaffold, cur_i: int, i_size: int, 
#     ct_added: int, ct_to_add: int, 
#     first_instance: bool, cap: int
# ) -> Generator[IndexingScaffold, None, None]:
#     if ct_to_add > s.capacity:
#         return
#     if s.n == 1:
#         yield from iter_variations_(
#             s.template, cur_i, i_size, ct_added, ct_to_add, first_instance)
#         return
#     if ct_to_add == 0:
#         yield s
#         return
#     min_ct = (ct_to_add - 1) // s.n
#     # print(s, min_ct, cap)
#     for ct in range(cap, min_ct, -1):
#         for lhs in iter_variations_(
#             s.template, cur_i, i_size, ct_added, ct, first_instance
#         ):
#             if first_instance:
#                 lhs = block_size(lhs, (i_size, ct_added + ct))
#             # don't need to check ct == 0 because ct > min_ct >= 0
#             for rhs in iter_variations_repeat_(
#                 RepeatScaffold(s.template, s.n-1),
#                 cur_i, i_size, ct_added + ct, ct_to_add - ct, False, ct
#             ):
#                 yield ProductScaffold(lhs, rhs)


# def unique_indexings_(
#     s: IndexingScaffold, index_types: 'list[tuple(int, int)]',
#     cur_i_type: int, cur_type_ct: int, max_i: int, cur_i: int
# ) -> Generator[IndexingScaffold, None, None]:
#     if cur_i == max_i:
#         yield s
#         return
#     if cur_type_ct == index_types[cur_i_type][1]:
#         cur_i_type += 1
#         cur_type_ct = 0

#     cur_i_size = index_types[cur_i_type][0]
#     #print("=<", s, "<<", cur_i)
#     for v in iter_variations_(s, cur_i, cur_i_size, 0, cur_i_size, True):
#         yield from unique_indexings_(
#             v, index_types, cur_i_type, cur_type_ct + 1,
#             max_i, cur_i + 1)
#     #print(">=", s, "<<", cur_i)

# def unique_indexings(
#     s: IndexingScaffold, index_types: 'list[tuple(int, int)]'
# ) -> Generator[IndexingScaffold, None, None]:
#     max_i = sum(type_ct for _, type_ct in index_types)
#     yield from unique_indexings_(s, index_types, 0, 0, max_i, 0)
    
# def canonicalize(s: IndexingScaffold, canon):
#     # this function should actually work
#     if isinstance(s, Index):
#         if s.value <= 0:
#             return s
#         canon[s.value] = canon.get(s.value, len(canon) + 1)
#         return Index(canon[s.value])
#     elif isinstance(s, RepeatScaffold):
#         return RepeatScaffold(canonicalize(s.template, canon), s.n)
#     elif isinstance(s, ProductScaffold):
#         lhs = canonicalize(s.l, canon)
#         return ProductScaffold(lhs, canonicalize(s.r, canon))