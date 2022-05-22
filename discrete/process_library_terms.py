import numpy as np
import copy
from findiff import FinDiff

from library import *
from weight import *
from convolution import *


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
        return max(self.line_dist(coord, i) for i, coord in enumerate(pt))
        # return np.linalg.norm([self.line_dist(coord, dim) for i, coord in enumerate(pt)])

    def line_dist(self, coord, dim):
        return max(0, self.min_corner[dim] - coord, coord - self.max_corner[dim])


# class CGPIndexing(object): # a choice of indexing within coarse-grained primitive
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


def lists_for_N(nloops, loop_max):
    if nloops == 0:
        yield []
        return
    for i in range(loop_max + 1):
        for li in lists_for_N(nloops - 1, loop_max):
            yield [i] + li


class SRDataset(object):  # structures all data associated with a given sparse regression dataset
    def __init__(self, world_size, data_dict, particle_pos, observables, kernel_sigma, cg_res, deltat, cutoff=8):
        self.world_size = world_size  # linear dimensions of dataset
        self.n_dimensions = len(world_size)  # number of dimensions (spatial + temporal)
        self.data_dict = data_dict  # observable -> array of values (particle, spatial index, time)
        self.cg_dict = dict()  # storage of computed coarse-grained quantities
        self.cgps = []  # list of coarse-grained primitives involved
        self.particle_pos = particle_pos  # array of particle positions (particle, spatial index, time)
        self.observables = observables  # list of observables
        self.kernel_sigma = kernel_sigma  # standard deviation of kernel in physical units (scalar for now)
        self.cg_res = cg_res  # subsampling factor when computing coarse-graining, i.e. cg_res points per unit length;
        # should generally just be an integer
        self.scaled_sigma = kernel_sigma * cg_res
        self.scaled_pts = particle_pos * cg_res
        self.dxs = [1 / cg_res] * (self.n_dimensions - 1) + [deltat]  # spacings of sampling grid
        self.domain_neighbors = None
        self.cutoff = cutoff  # how many std deviations to cut off gaussians at
        self.scale_dict = None  # dict of characteristic scales of observables -> (mean, std)

    def make_libraries(self, max_complexity=4, max_observables=3):
        self.libs = dict()
        terms = generate_terms_to(max_complexity, observables=self.observables,
                                  max_observables=max_observables)
        for rank in (0, 1):
            self.libs[rank] = LibraryData([term for term in terms if term.rank == rank], rank)

    def make_domains(self, ndomains, domain_size, pad=0):
        self.domains = []
        scaled_dims = [int(s * self.cg_res) for s in domain_size[:-1]] + [domain_size[-1]]
        scaled_world_size = [int(s * self.cg_res) for s in self.world_size[:-1]] + [self.world_size[-1]]
        # scaled_pad = np.ceil(pad*self.cg_res)
        self.domain_size = scaled_dims
        for i in range(ndomains):
            min_corner = []
            max_corner = []
            # define domains on the *scaled* grid
            for (L, max_lim) in zip(scaled_dims, scaled_world_size):
                num = np.random.randint(pad, max_lim - (L + pad) + 1)
                min_corner.append(num)
                max_corner.append(num + L - 1)
            self.domains.append(IntegrationDomain(min_corner, max_corner))

    def find_domain_neighbors(self):
        # list of indices corresponding to particles needed to compute quantities on each domain at each t
        self.domain_neighbors = dict()
        for domain in self.domains:
            for t in domain.times:
                self.domain_neighbors[domain, t] = []
                for i, pt in enumerate(self.scaled_pts):
                    dist = domain.distance(pt[:, t])
                    if dist <= self.scaled_sigma * self.cutoff:
                        self.domain_neighbors[domain, t].append(i)

    def make_weights(self, m, qmax):
        self.weights = []
        self.weight_dxs = [(width - 1) / 2 * dx for width, dx in zip(self.domain_size, self.dxs)]
        for q in lists_for_N(self.n_dimensions, qmax):
            self.weights.append(Weight([m] * self.n_dimensions, q, [0] * self.n_dimensions,
                                       dxs=self.weight_dxs))

    def eval_term(self, term, weight, domain, debug):
        # term: IndexedTerm
        # weight
        # domain: IntegrationDomain corresponding to where the term is evaluated
        # return the evaluated term on the domain grid
        product = np.ones(shape=domain.shape)
        if debug:
            print(f"IndexedTerm {term}")
        for idx, prim in enumerate(term.obs_list):
            if debug:
                print(f"IndexedPrimitive {prim}, CGP {prim.cgp}")
            dorders = prim.dimorders
            obs_dims = prim.obs_dims
            cgp = prim.cgp
            if debug:
                print("dorders", dorders, "obs_dims", obs_dims)
            if (cgp, tuple(obs_dims), domain) in self.cg_dict.keys():  # field is "cached"
                data_slice = self.cg_dict[cgp, tuple(obs_dims), domain]
            else:
                data_slice = self.eval_cgp(cgp, obs_dims, domain)
                self.cg_dict[cgp, tuple(obs_dims), domain] = data_slice
                if cgp not in self.cgps:
                    if debug:
                        print(f"CGP {cgp} is new")
                    self.cgps.append(cgp)
            if sum(dorders) != 0:
                product *= diff(data_slice, dorders, self.weight_dxs)
            else:
                product *= data_slice
            # print(product[0, 0, 0])
        weight_arr = weight.get_weight_array(domain.shape)
        product *= weight_arr
        return product

    # TODO: implement coarse-graining
    def eval_cgp(self, cgp, obs_dims, domain):
        data_slice = np.zeros(domain.shape)
        if self.domain_neighbors is None:
            self.find_domain_neighbors()
        for t in range(domain.shape[-1]):
            time_slice = np.zeros(domain.shape[:-1])
            t_shifted = t + domain.min_corner[-1]
            for i in self.domain_neighbors[domain, t_shifted]:
                pt_pos = self.scaled_pts[i, :, t_shifted]
                # evaluate observables inside rho[...]
                coeff = 1
                obs_dim_ind = 0
                for obs in cgp.obs_list:
                    # print(obs, i, obs_dims[obs_dim_ind], t_shifted)
                    if obs.rank == 0:
                        coeff *= self.data_dict[obs.string][i, 0, t_shifted]
                    else:
                        coeff *= self.data_dict[obs.string][i, obs_dims[obs_dim_ind], t_shifted]
                    obs_dim_ind += obs.rank
                    # print(coeff)
                # coarse-graining this particle (one dimension at a time)
                rngs = []
                g_nd = 1
                for coord, d_min, d_max, i in zip(pt_pos, domain.min_corner, domain.max_corner,
                                                  range(self.n_dimensions - 1)):
                    # recenter so that 0 is start of domain
                    g, mn, mx = gauss1d(coord - d_min, self.scaled_sigma, truncate=self.cutoff,
                                        xmin=0, xmax=d_max - d_min)
                    g_nd = np.multiply.outer(g_nd, g)
                    rng_array = np.array(range(mn, mx))  # coordinate range of kernel
                    # now need to add free axes so that the index ends up as an (n-1)-d array
                    n_free_dims = self.n_dimensions - i - 2  # how many np.newaxis to add to index
                    expanded_rng_array = np.expand_dims(rng_array, axis=tuple(range(1, 1 + n_free_dims)))
                    rngs.append(expanded_rng_array)
                # if len((g_nd*coeff).shape) > len(time_slice.shape):
                #    print(rngs, g_nd.shape, coeff)
                time_slice[tuple(rngs)] += g_nd * coeff
            data_slice[..., t] = time_slice
        data_slice *= self.cg_res ** (self.n_dimensions - 1)  # need to scale rho by res^(# spatial dims)!
        return data_slice

    def make_library_matrices(self, by_parts=True, debug=False):
        for rank in (0, 1):
            if debug:
                print(f"***RANK {rank} LIBRARY***")
            terms = self.libs[rank].terms
            dshape = self.domain_size
            if rank == 1:
                d = len(dshape) - 1  # dimensionality of equation
            else:
                d = 1
            q = np.zeros(shape=(len(self.weights) * len(self.domains) * d, len(terms)))
            for i, term in enumerate(terms):
                row_index = 0
                if debug:
                    print("\ni:", i)
                # this check should be redundant now
                # if term.rank != rank: # select only terms of the desired rank
                #    continue
                if debug:
                    print("UNINDEXED TERM:")
                    print(term)
                for weight in self.weights:
                    for k in range(d):
                        if rank == 0:
                            kc = None
                        else:
                            kc = k
                        arr = np.zeros(np.append(dshape, len(self.domains)))
                        if isinstance(term, ConstantTerm):
                            # "short circuit" the evaluation to deal with constant term case
                            for p, domain in enumerate(self.domains):
                                weight_arr = weight.get_weight_array(dshape)
                                # I think this should not be weight_dxs?
                                q[row_index, i] = int_arr(weight_arr, self.dxs)
                                if debug and p == 0:
                                    print("Value: ", q[row_index, i])
                                row_index += 1
                            continue
                        # note: temporal index not included here
                        for (space_orders, obs_dims) in get_dims(term, len(dshape) - 1, kc):
                            # print(term, kc, list(get_dims(term, len(dshape)-1, kc)), space_orders, obs_dims)
                            # first, make labeling canonical within each CGP
                            if space_orders is None and obs_dims is None:
                                nt = len(term.obs_list)
                                space_orders = [[0] * len(dshape) for i in nt]
                                canon_obs_dims = [[None] * i.cgp.rank for i in nt]
                            else:
                                canon_obs_dims = []
                                for sub_list, prim in zip(obs_dims, term.obs_list):
                                    canon_obs_dims.append(prim.cgp.index_canon(sub_list))
                            # integrate by parts
                            indexed_term = IndexedTerm(term, space_orders, canon_obs_dims)
                            # note that we have taken integration by parts outside of the domain loop
                            if debug:
                                print("ORIGINAL TERM:")
                                print(indexed_term, [o.dimorders for o in indexed_term.obs_list])
                            if by_parts:
                                # integration by parts
                                for mod_term, mod_weight in int_by_parts(indexed_term, weight):
                                    if debug:
                                        print("INTEGRATED BY PARTS:")
                                        print(mod_term, [o.dimorders for o in mod_term.obs_list],
                                              mod_weight)
                                    for p, domain in enumerate(self.domains):
                                        arr[..., p] += self.eval_term(mod_term, mod_weight,
                                                                      domain, debug=(debug and p == 0))
                            else:
                                for p, domain in enumerate(self.domains):
                                    arr[..., p] += self.eval_term(indexed_term, weight, domain,
                                                                  debug=(debug and p == 0))
                        for p in range(len(self.domains)):
                            q[row_index, i] = int_arr(arr[..., p], self.dxs)
                            if debug and p == 0:
                                print("Value: ", q[row_index, i])
                            row_index += 1
            self.libs[rank].Q = q
            # return q

            # this is also a reasonable place to compute the column weights
            # if self.scale_dict is None: # haven't computed scales yet
        self.find_scales()
        for rank in (0, 1):
            self.libs[rank].col_weights = [self.get_char_size(term) for term in self.libs[rank].terms]
            # if rank==0: # then we can also compute the row weights
        self.find_row_weights()

    # TODO - just take empirical average/std over points
    def find_scales(self, names=None):
        # find mean/std deviation of fields in data_dict that are in names
        self.scale_dict = dict()
        for name in self.data_dict:
            if names is None or name in names:
                self.scale_dict[name] = dict()
                # if these are vector quantities the results could be wonky in the unlikely
                # case a vector field is consistently aligned with one of the axes
                self.scale_dict[name]['mean'] = np.mean(np.linalg.norm(self.data_dict[name]))
                self.scale_dict[name]['std'] = np.std(self.data_dict[name])
        # also need to handle density separately
        self.scale_dict['rho'] = dict()
        self.scale_dict['rho']['mean'] = self.particle_pos.shape[0] / np.prod(self.world_size)
        rho_ind = find_term(self.cgps, 'rho')
        rho_cgp = self.cgps[rho_ind]
        rho_std = np.std(np.dstack([self.cg_dict[rho_cgp, (), domain] for domain in self.domains]))
        self.scale_dict['rho']['std'] = rho_std

    # TODO - needs to be rewritten a bit
    def get_char_size(self, term):
        # return characteristic size of a library term
        product = 1
        for prim in term.obs_list:
            xorder = prim.dorder.xorder
            torder = prim.dorder.torder
            if torder + xorder > 0:
                statistic = 'std'
            else:
                statistic = 'mean'
            for obs in prim.cgp.obs_list:
                name = obs.string
                product *= self.scale_dict[name][statistic]
            # add in rho contribution (every primitive contains a rho)
            product *= self.scale_dict['rho'][statistic]
            # scale by grid spacings for derivatives
            product *= self.dxs[0] ** xorder
            product *= self.dxs[-1] ** torder
        return product

    def find_row_weights(self):
        rho_col = find_term(self.libs[0].terms, 'rho')
        # integral of rho with the 0 harmonics weight
        dom_densities = self.libs[0].Q[0:len(self.domains), rho_col]
        row_weights0 = np.tile(dom_densities, len(self.weights))
        # scale weights according to square root of density (to cancel CLT noise scaling)
        row_weights0 = np.sqrt(row_weights0)
        row_weights0 += 1e-6  # don't want it to be exactly zero
        # normalize
        row_weights0 /= np.max(row_weights0)
        row_weights1 = np.tile(row_weights0, self.n_dimensions - 1)  # because each dimension gets its own row
        self.libs[0].row_weights = row_weights0
        self.libs[1].row_weights = row_weights1


# this class might be absorbed into SRDataset
class LibraryData(object):  # structures information associated with a given rank library
    def __init__(self, terms, rank):  # , parent
        # self.parent = parent
        self.terms = terms
        self.rank = rank
        self.Q = None  # Q matrix
        self.col_weights = None
        self.row_weights = None


def find_term(term_list, string):  # find index of term in list matching string
    return [str(elt) for elt in term_list].index(string)


def get_slice(arr, domain):
    arr_slice = arr
    for (slice_dim, min_c, max_c) in zip(range(arr.ndim), domain.min_corner, domain.max_corner):
        idx = [slice(None)] * arr.ndim
        idx[slice_dim] = slice(min_c, max_c + 1)
        arr_slice = arr_slice[tuple(idx)]
    return arr_slice


# FIXME UNTESTED
def get_dims(term, ndims, dim=None, start=0, do=None, od=None):
    # yield all of the possible x, y, z labelings for a given LibraryTerm
    labels = term.labels
    # print(term, labels, dim)
    if do is None:
        do = [[0] * ndims for _ in term.obs_list]  # derivatives in x, y (z) of each part of the term
    if od is None:
        od = [[None] * t.cgp.rank for t in
              term.obs_list]  # the dimension of the observable to evaluate (None if the observable is rank 0)
    if len(labels.keys()) == 0:
        yield do, od
        return
    if start == 0:
        start += 1
        if dim is not None:
            val = labels[0][0]
            if val[0] % 2 == 0:
                do[val[0] // 2][dim] += 1
            else:
                od[val[0] // 2][val[1]] = dim
    if start > max(labels.keys()):
        yield do, od
    else:
        vals = labels[start]
        for new_dim in range(ndims):
            do_new = copy.deepcopy(do)
            od_new = copy.deepcopy(od)
            # print("do", do)
            # print("od", od)
            for val_ind, val_pos in vals:
                if val_ind % 2 == 0:
                    do_new[val_ind // 2][new_dim] += 1
                else:
                    # noinspection PyTypeChecker
                    od_new[val_ind // 2][val_pos] = new_dim
            # print("do_new", do_new)
            # print("od_new", od_new)
            yield from get_dims(term, ndims, dim=dim, start=start + 1, do=do_new, od=od_new)


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
    best_i, next_i = None, None
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
        elif prim.nderivs == term.nderivs - 1 and next_prim is None:
            next_i, next_prim = i, prim
        # elif prim.nderivs == term.nderivs-2:
        #    third_i, third_best = i, prim
    # check viability by cases
    newords = best_prim.dimorders.copy()
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
        rest = rest.drop(next_prim)
        if next_prim.succeeds(best_prim, dim):  # check if next_best goes with best
            # check if next best is unique
            for obs in rest.obs_list:
                if obs == next_prim:  # can stop here because we limit the number of terms
                    # x' * x * x case
                    # print(rest, next_prim)
                    rest = rest.drop(obs)
                    for summand in rest.diff(dim):
                        yield next_prim * next_prim * next_prim * summand, -1 / 3 * weight, False
                    yield next_prim * next_prim * next_prim * rest, -1 / 3 * new_weight, False
                    return
                elif obs.nderivs == term.nderivs - 1:  # not unique and doesn't match
                    yield term, weight, True
                    return
            # x' * x case
            for summand in rest.diff(dim):
                yield next_prim * next_prim * summand, -1 / 2 * weight, False
            yield next_prim * next_prim * rest, -1 / 2 * new_weight, False
            return
        else:
            yield term, weight, True
            return


def diff(data, dorders, dxs=None, acc=4):
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

# cache the derivatives through order 2? fill in code below for using cache
# def encode(obs_name, dorders):
#    # e.g. dxx
#    return ""
#
# def decode(data_name):
#    # return obs_name, dorders
#    return "", [0, 0, 0]
