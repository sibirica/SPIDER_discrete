from findiff import FinDiff
from commons.weight import *
from library import *
from functools import reduce
from operator import add, mul


class IntegrationDomain(object):
    def __init__(self, min_corner, max_corner):
        # min_corner - min coordinates in each dimension; sim. max_corner
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.shape = [max_c - min_c + 1 for (min_c, max_c) in zip(self.min_corner, self.max_corner)]

    def __repr__(self):
        return f"IntegrationDomain({self.min_corner}, {self.max_corner})"


class Weight(object):
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


def make_domains(dims, world_size, ndomains, pad=0):
    domains = []
    for i in range(ndomains):
        min_corner = []
        max_corner = []
        for (L, max_lim) in zip(dims, world_size):
            num = np.random.randint(pad, max_lim - (L + pad) + 1)
            min_corner.append(num)
            max_corner.append(num + L - 1)
        domains.append(IntegrationDomain(min_corner, max_corner))
    return domains


def get_slice(arr, domain):
    arr_slice = arr
    for (slice_dim, min_c, max_c) in zip(range(arr.ndim), domain.min_corner, domain.max_corner):
        idx = [slice(None)] * arr.ndim
        idx[slice_dim] = slice(min_c, max_c + 1)
        arr_slice = arr_slice[tuple(idx)]
    return arr_slice


# integrate by parts and specify weight functions
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


def int_by_parts_dim(term, weight, dim):
    """for integration by parts, check terms that look like x', x*x', and x*x*x' (vs all other terms have derivative
    orders smaller by at least 2) this misses out on opportunities to integrate by parts using a different basis, but
    this seems too difficult to automate; at this point it's probably easier to just write the term out manually."""

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


def diff(data, dorders, dxs=None):
    # for spatial directions can use finite differences or spectral differentiation. For time, only the former.
    # in any case, it's probably best to pre-compute the derivatives on the whole domains (at least up to order 2).
    # with integration by parts, there shouldn't be higher derivatives.
    acc = 6  # order of accuracy that we want
    if dxs is None:
        dxs = [1] * len(dorders)
    diff_list = []
    for i, dx, order in zip(range(len(dxs)), dxs, dorders):
        if order > 0:
            diff_list.append((i, dx, order))
    diff_operator = FinDiff(*diff_list, acc=acc)
    return diff_operator(data)


# we want to cache all derivatives as they are computed, so we want to be able to encode+decode observable name and the
# derivative orders

def encode(obs_name, dorders):
    # technically if you could go to 10th derivative you'd want to add commas between the numbers
    if sum(dorders) == 0:
        return obs_name
    return obs_name + "~" + reduce(add, [str(do) for do in dorders])


def decode(data_name):  # this function is not strictly necessary
    str_list = data_name.split("~")
    return str_list[0], [int(c) for c in str_list[1]]


def eval_term(lt, weight, data_dict, domain, dxs, debug=False):
    # lt: LibraryTerm
    # weight
    # data_dict: keys are Observable names, values are data arrays
    # domain: IntegrationDomain corresponding to where the term is evaluated
    # return the evaluated term on the domain grid

    product = np.ones(shape=domain.shape)
    # won't execute at all for a constant term
    if debug:
        print(f"LibraryTerm {lt}")
    for idx, obs in enumerate(lt.obs_list):
        dorders = obs.dimorders
        obs_dim = obs.obs_dim
        # if debug:
        #    print("dorders", dorders, "obs_dim", obs_dim)
        name = obs.observable.string
        en_name = encode(name, dorders)
        # print(en_name)
        if en_name in data_dict and (obs_dim is None or obs_dim < data_dict[en_name].shape[-1]):  # field is "cached"
            if obs_dim is None:
                data_arr = data_dict[en_name]
            else:
                data_arr = data_dict[en_name][..., obs_dim]
            product *= get_slice(data_arr, domain)
        else:
            if obs_dim is None:
                data_arr = data_dict[name]
            else:
                data_arr = data_dict[name][..., obs_dim]
            data_slice = get_slice(data_arr, domain)
            if sum(dorders) != 0:
                product *= diff(data_slice, dorders, dxs)
                # full_diff = diff(data_arr, dorders, dxs)
                # product *= get_slice(full_diff, domain)
                # if obs_dim is None:
                #    data_dict[name] = data_arr
                # else:
                #    if en_name not in data_dict:
                #        data_dict[name] = np.zeros(shape=data_arr[..., np.newaxis].shape)
                #    data_dict[name][..., obs_dim] = data_arr
            else:
                product *= data_slice
        # print(product[0, 0, 0])
    weight_arr = weight.get_weight_array(domain.shape)
    product *= weight_arr
    return product


def get_dims(term, ndims, dim=None, start=0, do=None, od=None):
    # yield all of the possible x, y, z labelings for a given term
    labels = term.labels
    # print(labels)
    if do is None:
        do = [[0] * ndims for _ in term.obs_list]  # derivatives in x, y (z) of each part of the term
    if od is None:
        od = [None] * len(
            term.obs_list)  # the dimension of the observable to evaluate (None if the observable is rank 0)
    if len(labels.keys()) == 0:
        yield do, od
        return
    if start == 0:
        start += 1
        if dim is not None:
            val = labels[0][0]
            if val % 2 == 0:
                do[val // 2][dim] += 1
            else:
                od[val // 2] = dim
    if start > max(labels.keys()):
        yield do, od
    else:
        vals = labels[start]
        for new_dim in range(ndims):
            do_new = copy.deepcopy(do)
            od_new = copy.deepcopy(od)
            for val in vals:
                if val % 2 == 0:
                    do_new[val // 2][new_dim] += 1
                else:
                    # noinspection PyTypeChecker
                    od_new[val // 2] = new_dim
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


def make_library(terms, data_dict, weights, domains, rank, dxs=None, by_parts=True, debug=False):
    dshape = domains[0].shape
    if debug:
        print(f"***RANK {rank} LIBRARY***")
    if rank == 1:
        d = len(dshape) - 1  # dimensionality of equation
    else:
        d = 1
    q = np.zeros(shape=(len(weights) * len(domains) * d, len(terms)))
    for i, term in enumerate(terms):
        row_index = 0
        if debug:
            print("\ni:", i)
        if term.rank != rank:  # select only terms of the desired rank
            continue
        if debug:
            print("UNINDEXED TERM:")
            print(term)
        for weight in weights:
            for k in range(d):
                if rank == 0:
                    kc = None
                else:
                    kc = k
                arr = np.zeros(np.append(dshape, len(domains)))
                if isinstance(term, ConstantTerm):
                    # "short circuit" the evaluation to deal with constant term case
                    for _p, domain in enumerate(domains):
                        arr[..., _p] = eval_term(term, weight, data_dict, domain, dxs, debug=(debug and _p == 0))
                        q[row_index, i] = int_arr(arr[..., _p], dxs)
                        if debug and _p == 0:
                            print("Value: ", q[row_index, i])
                        row_index += 1
                    continue
                for (space_orders, obs_dims) in get_dims(term, len(dshape) - 1,
                                                         kc):  # note: temporal index not included here
                    if space_orders is None and obs_dims is None:
                        nt = len(term.obs_list)
                        space_orders = [[0] * len(dshape) for i in range(nt)]
                        obs_dims = [None] * nt
                    # integrate by parts
                    indexed_term = IndexedTerm(term, space_orders, obs_dims)
                    # note that we have taken integration by parts outside of the domain loop
                    if debug:
                        print("ORIGINAL TERM:")
                        print(indexed_term, [o.dimorders for o in indexed_term.obs_list])
                    if by_parts:
                        for mod_term, mod_weight in int_by_parts(indexed_term, weight):  # integration by parts
                            if debug:
                                print("INTEGRATED BY PARTS:")
                                print(mod_term, [o.dimorders for o in mod_term.obs_list], mod_weight)
                            for _p, domain in enumerate(domains):
                                arr[..., _p] += eval_term(mod_term, mod_weight, data_dict, domain, dxs,
                                                          debug=(debug and _p == 0))
                    else:
                        for _p, domain in enumerate(domains):
                            arr[..., _p] += eval_term(indexed_term, weight, data_dict, domain, dxs,
                                                      debug=(debug and _p == 0))

                for _p in range(len(domains)):
                    q[row_index, i] = int_arr(arr[..., _p], dxs)
                    if debug and _p == 0:
                        print("Value: ", q[row_index, i])
                    row_index += 1
    return q


def find_scales(data_dict, names=None):
    # find mean/std deviation of fields in data_dict that are in names
    scale_dict = dict()
    for name in data_dict:
        if names is None or name in names:
            scale_dict[name] = dict()
            scale_dict[name]['mean'] = np.mean(np.abs(data_dict[name]))
            scale_dict[name]['std'] = np.std(data_dict[name])
    return scale_dict


def get_char_size(term, scale_dict, dx, dt):
    # return characteristic size of a library term
    product = 1
    for tm in term.obs_list:
        xorder = tm.dorder.xorder
        torder = tm.dorder.torder
        name = tm.observable.string
        if torder + xorder > 0:
            product *= scale_dict[name]['std']
        else:
            product *= scale_dict[name]['mean']
        # scale by grid spacings for derivatives (should already be accounted for by findiff)
        # product /= dx**xorder
        # product /= dt**torder
    return product if product > 0 else 1  # if the variable is always 0 then we'll get division by zero


def find_term(term_list, string):  # find index of term in list matching string
    return [str(elt) for elt in term_list].index(string)
