#from commons.weight import *
from commons.process_library_terms import *
#from commons.library import * # don't know if it's needed
from library import *


class SRDataset(AbstractDataset):
    #field_dict: dict[tuple[Any], np.ndarray[float]] = None # storage of computed coarse-grained quantities: (prim, dims, domains) -> array
    def make_domains(self, ndomains, domain_size, pad=0):
        self.domains = []
        self.domain_size = domain_size
        for i in range(ndomains):
            min_corner = []
            max_corner = []
            for (L, max_lim) in zip(domain_size, self.world_size):
                num = np.random.randint(pad, max_lim - (L + pad) + 1)
                min_corner.append(num)
                max_corner.append(num + L - 1)
            self.domains.append(IntegrationDomain(min_corner, max_corner))
        #return domains

    def eval_prim(self, prim, obs_dims, domain):
        if obs_dim is None:
            data_arr = self.data_dict[name]
        else:
            data_arr = self.data_dict[name][..., obs_dim]
        data_slice = get_slice(data_arr, domain)
        return data_slice
    
    def make_libraries(self, max_complexity=4, max_observables=3):
        self.libs = dict()
        terms = generate_terms_to(max_complexity, observables=self.observables,
                                  max_observables=max_observables)
        for rank in self.irreps:
            self.libs[rank] = LibraryData([term for term in terms if term.rank == rank], rank)

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

    @staticmethod
    def get_dims(term, ndims, dim=None, start=0, do=None, od=None): ## TO DO: REWORK FOR HIGHER-RANK CASE
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
