import numpy as np
from library import *
from weight import *
import copy
from findiff import FinDiff

class IntegrationDomain(object):
    def __init__(self, min_corner, max_corner):
        # min_corner - min coordinates in each dimension; sim. max_corner
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.shape = [max_c-min_c+1 for (min_c, max_c) in zip(self.min_corner, self.max_corner)]
        
    def __repr__(self):
        return f"IntegrationDomain({self.min_corner}, {self.max_corner})"
    
    def __hash__(self): # for use in cg_dict
        # here we use that min_corner and max_corner are both lists of ints
        return hash((tuple(self.min_corner+self.max_corner)))
    
#class CGPIndexing(object): # a choice of indexing within coarse-grained primitive
#    def __init__(self, index_list):
#        # min_corner - min coordinates in each dimension; sim. max_corner
#        self.index_list = index_list
#        
#    def __repr__(self):
#        return f"CGPIndexing({self.index_list})"
#    
#    def __hash__(self): # for use in cg_dict
#        # here we use that min_corner and max_corner are both lists of ints
#        return hash((tuple(self.index_list)))
    
class Weight(object):
    def __init__(self, m, q, k, scale=1, dxs=None):
        self.m = m
        self.q = q
        self.k = k
        self.scale = scale
        self.ready = False
        self.weight_obj = None
        if dxs is None:
            self.dxs = [1]*len(self.m)
        else:
            self.dxs = dxs
    
    def make_weight_objs(self):
        self.ready = True
        self.weight_objs = [weight_1d(m, q, k, dx) for (m, q, k, dx) in zip(self.m, self.q, self.k, self.dxs)]
    
    def get_weight_array(self, dims):
        if not self.ready:
            self.make_weight_objs()
        weights_eval = [weight.linspace(dim)[1] for (weight, dim) in zip(self.weight_objs, dims)]
        return self.scale*reduce(lambda x, y: np.tensordot(x, y, axes=0), weights_eval)
    
    def increment(self, dim): # return new weight with an extra derivative on the dim-th dimension
        knew = self.k.copy()
        knew[dim] += 1
        return Weight(self.m, self.q, knew, scale=self.scale, dxs=self.dxs)
    
    def __neg__(self):
        return Weight(self.m, self.q, self.k, scale=-self.scale, dxs=self.dxs)
    
    def __mul__(self, number):
        return Weight(self.m, self.q, self.k, scale=self.scale*number, dxs=self.dxs)
        
    __rmul__ = __mul__
    
    def __repr__(self):
        return f"Weight({self.m}, {self.q}, {self.k}, {self.scale}, {self.dxs})"
    
def make_domains(dims, world_size, ndomains, pad=0):
    domains = []
    for i in range(ndomains):
        min_corner = []
        max_corner = []
        for (L, max_lim) in zip(dims, world_size):
            num = np.random.randint(pad, max_lim-(L+pad)+1)
            min_corner.append(num)
            max_corner.append(num+L-1)
        domains.append(IntegrationDomain(min_corner, max_corner))
    return domains

def get_slice(arr, domain):
    arr_slice = arr
    for (slice_dim, min_c, max_c) in zip(range(arr.ndim), domain.min_corner, domain.max_corner):
        idx = [slice(None)]*arr.ndim
        idx[slice_dim] = slice(min_c, max_c+1)
        arr_slice = arr_slice[tuple(idx)]
    return arr_slice

### integrate by parts and specify weight functions ###
def int_by_parts(term, weight, dim=0):
    if dim>=term.ndims:
        yield term, weight
    else:
        failed = False
        for te, we, fail in int_by_parts_dim(term, weight, dim): # try to integrate by parts
            failed = (failed or fail)
            if failed: # can't continue, so go to next dimension
                dim += 1
            yield from int_by_parts(te, we, dim) # repeat process (possibly on next dimension)

# for integration by parts, check terms that look like x', x*x', and x*x*x' (vs all other terms have derivative orders smaller by at least 2)
# this misses out on opportunities to integrate by parts using a different basis, but this seems too difficult to automate; 
# at this point it's probably easier to just write the term out manually.
def int_by_parts_dim(term, weight, dim):
    # find best term to base integration off of
    #best_prim, next_best, third_best = None, None, None
    #best_i, next_i, third_i = None, None, None
    best_prim, next_prim = None, None
    best_i, next_i = None, None
    for (i, prim) in enumerate(term.observable_list):
        if prim.nderivs == term.nderivs:
            if best_prim is None:
                best_i, best_prim = i, prim
                if prim.dimorders[dim]==0: # can't integrate by parts along this dimension
                    yield term, weight, True
                    return 
            else: # multiple candidate terms -> integrating by parts will not help
                yield term, weight, True
                return  
        elif prim.nderivs == term.nderivs-1 and next_prim is None:
            next_i, next_prim = i, prim
        #elif prim.nderivs == term.nderivs-2:
        #    third_i, third_best = i, prim
    # check viability by cases
    newords = best_prim.dimorders.copy()
    newords[dim] -= 1
    new_weight = weight.increment(dim)
    new_prim = IndexedPrimitive(best_prim, newords=newords)
    #print(term, best_prim)
    rest = term.drop(best_prim)
    if next_prim is None: # then all other terms have derivatives up to order n-2, so we are in x' case
        for summand in rest.diff(dim):
            yield new_prim*summand, -weight, False
        yield new_prim*rest, -new_weight, False
        return
    else:
        #print(rest, next_prim)
        rest = rest.drop(next_prim)
        if next_prim.succeeds(best_prim, dim): # check if next_best goes with best
            # check if next best is unique
            for obs in rest.observable_list:
                if obs == next_prim: # can stop here because we limit the number of terms
                    # x' * x * x case
                    #print(rest, next_prim)
                    rest = rest.drop(obs)
                    for summand in rest.diff(dim):
                        yield next_prim*next_prim*next_prim*summand, -1/3*weight, False
                    yield next_prim*next_prim*next_prim*rest, -1/3*new_weight, False
                    return
                elif obs.nderivs == term.nderivs-1: # not unique and doesn't match
                    yield term, weight, True
                    return
            # x' * x case
            for summand in rest.diff(dim):
                yield next_prim*next_prim*summand, -1/2*weight, False
            yield next_prim*next_prim*rest, -1/2*new_weight, False
            return
        else:
            yield term, weight, True
            return

def diff(data, dorders, dxs=None):
    # for spatial directions can use finite differences or spectral differentiation. For time, only the former.
    # in any case, it's probably best to pre-compute the derivatives on the whole domains (at least up to order 2). with integration by parts, there shouldn't be higher derivatives.
    acc = 6 # order of accuracy that we want
    if dxs is None:
        dxs = [1]*len(dorders)
    diff_list = []
    for i, dx, order in zip(range(len(dxs)), dxs, dorders):
        if order>0:
            diff_list.append((i, dx, order))
    diff_operator = FinDiff(*diff_list, acc=acc)
    return diff_operator(data)

# cache the derivatives through order 2? fill in code below for using cache
#def encode(obs_name, dorders):
#    # e.g. dxx
#    return ""
#
#def decode(data_name):
#    # return obs_name, dorders
#    return "", [0, 0, 0]

def eval_term(lt, weight, scaled_pts, data_dict, cg_dict, domain, dxs, kernel_dxs, debug=False): #, dim
    # lt: LibraryTerm
    # weight
    # pos: (scaled) positions of data points
    # data_dict: keys are Observable names, values are data arrays (per point in pos)
    # domain: IntegrationDomain corresponding to where the term is evaluated
    # dim: dimension of the vector to be returned (e.g., 0, 1, 2, or None)
    # return the evaluated term on the domain grid

    product = np.ones(shape=domain.shape)
    # wouldn't execute at all for a constant (1) term
    # TO DO: add logic for rho[1]
    if debug:
        print(f"LibraryTerm {lt}")
    for idx, prim in enumerate(lt.observable_list):
        dorders = prim.dimorders
        obs_dims = prim.obs_dims
        name = prim.observable.string

        #indexing = CGPIndexing(obs_dims) # we'll see if this class is useful
        cgp = prim.cgp
        #if debug:
        #    print("dorders", dorders, "obs_dims", obs_dims)
        if (cgp, tuple(obs_dims), domain) in cg_dict.keys(): # field is "cached"
            data_slice = cg_dict[cgp, tuple(obs_dims), domain]
            product *= data_slice
        else:
            data_slice = eval_cgp(cgp, obs_dims, domain)
            cg_dict[cgp, tuple(obs_dims), domain] = data_slice
            if sum(dorders)!=0:
                product *= diff(data_slice, dorders, dxs)
            else:
                product *= data_slice
        #print(product[0, 0, 0])
    weight_arr = weight.get_weight_array(domain.shape)
    product *= weight_arr
    return product

## TO DO: implement coarse-graining
def eval_cgp():
    ###

### UNTESTED ###
def get_dims(term, ndims, dim=None, start=0, do=None, od=None):
    # yield all of the possible x, y, z labelings for a given term
    labels = term.labels
    #print(labels)
    if do is None:
        do = [[0]*ndims for t in term.observable_list] # derivatives in x, y (z) of each part of the term
    if od is None:
        od = [[None]*t.cgp.rank for t in term.observable_list] # the dimension of the observable to evaluate (None if the observable is rank 0)
    if len(labels.keys())==0:
        yield do, od
        return
    if start==0:
        start += 1
        if dim is not None:
            val = labels[0][0]
            if val%2==0:
                do[val[0]//2][dim] += 1
            else:
                od[val[0]//2][val[1]] = dim
    if start>max(labels.keys()):
        yield do, od
    else:
        vals = labels[start]
        for new_dim in range(ndims):
            do_new = copy.deepcopy(do)
            od_new = copy.deepcopy(od)
            #print("do", do)
            #print("od", od)
            for val_ind, val_pos in vals:
                if val%2==0:
                    do_new[val_ind//2][new_dim] += 1
                else:
                    od_new[val_ind//2][val_pos] = new_dim
            #print("do_new", do_new)
            #print("od_new", od_new)
            yield from get_dims(term, ndims, dim=dim, start=start+1, do=do_new, od=od_new) 

def int_arr(arr, dxs=None): # integrate the output of eval_term with respect to weight function
    if dxs is None:
        dxs = [1]*len(arr.shape)
    dx = dxs[0]
    integral = np.trapz(arr, axis=0)
    if len(dxs)==1:
        return integral
    else:
        return int_arr(integral, dxs[1:])

def make_library(terms, data_dict, cg_dict, weights, domains, rank, dxs=None, by_parts=True, debug=False):
    dshape = domains[0].shape
    if debug:
        print(f"***RANK {rank} LIBRARY***")
    if rank==1:
        d = len(dshape)-1 # dimensionality of equation
    else:
        d = 1
    Q = np.zeros(shape=(len(weights)*len(domains)*d, len(terms)))
    for i, term in enumerate(terms):
        row_index = 0
        if debug:
            print("\ni:", i)
        if term.rank != rank: # select only terms of the desired rank
            continue
        if debug:
            print("UNINDEXED TERM:")
            print(term)
        for weight in weights:
            for k in range(d):
                if rank==0:
                    kc = None
                else:
                    kc = k
                arr = np.zeros(np.append(dshape, len(domains)))
                if isinstance(term, ConstantTerm):
                    # "short circuit" the evaluation to deal with constant term case
                    for p, domain in enumerate(domains):
                        arr[..., p] = eval_term(term, weight, data_dict, cg_dict, domain, dxs, debug=(debug and p==0))
                        Q[row_index, i] = int_arr(arr[..., p], dxs)
                        if debug and p==0:
                            print("Value: ", Q[row_index, i])
                        row_index += 1
                    continue
                for (space_orders, obs_dims) in get_dims(term, len(dshape)-1, kc): # note: temporal index not included here
                    # first, make labeling canonical within each CGP
                    if space_orders is None and obs_dims is None:
                        nt = len(term.observable_list)
                        space_orders = [[0]*len(dshape) for i in nt]
                        canon_obs_dims = [[None]*i.cgp.rank for i in nt]
                    else:
                        canon_obs_dims = []
                        for sub_list, prim in zip(obs_dims, term.observable_list):
                            canon_obs_dims.append(prim.cgp.index_canon(sub_list))
                    # integrate by parts
                    indexed_term = IndexedTerm(term, space_orders, canon_obs_dims)
                    # note that we have taken integration by parts outside of the domain loop
                    if debug:
                        print("ORIGINAL TERM:")
                        print(indexed_term, [o.dimorders for o in indexed_term.observable_list])
                    if by_parts:
                        for mod_term, mod_weight in int_by_parts(indexed_term, weight): # integration by parts
                            if debug:
                                print("INTEGRATED BY PARTS:")
                                print(mod_term, [o.dimorders for o in mod_term.observable_list], mod_weight)
                            for p, domain in enumerate(domains):
                                arr[..., p] += eval_term(mod_term, mod_weight, data_dict, cg_dict, domain, dxs, 
                                                         debug=(debug and p==0))
                    else:
                        for p, domain in enumerate(domains):
                            arr[..., p] += eval_term(indexed_term, weight, data_dict, cg_dict, domain, dxs, debug=(debug and p==0))

                for p in range(len(domains)):
                    Q[row_index, i] = int_arr(arr[..., p], dxs)
                    if debug and p==0:
                        print("Value: ", Q[row_index, i])
                    row_index += 1
    return Q


## TO DO - rewrite to work with cg_dict
def find_scales(cg_dict, domains, names=None): 
    # find mean/std deviation of fields in data_dict that are in names
    scale_dict = dict()
    for name in cg_dict:
        if names is None or name in names:
            scale_dict[name] = dict()
            scale_dict[name]['mean'] = np.mean(np.abs(data_dict[name]))
            scale_dict[name]['std'] = np.std(data_dict[name])
    return scale_dict
    
def get_char_size(term, scale_dict, dx, dt):
    # return characteristic size of a library term
    product = 1
    for tm in term.observable_list:
        xorder = tm.dorder.xorder
        torder = tm.dorder.torder
        name = tm.observable.string
        if torder+xorder>0:
            product *= scale_dict[name]['mean']
        else:
            product *= scale_dict[name]['std']
        product *= dx**xorder
        product *= dt**torder
    return product

def find_term(term_list, string): # find index of term in list matching string
    return [str(elt) for elt in term_list].index(string)