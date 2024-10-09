#from commons.weight import *
from commons.process_library_terms import *
#from commons.library import * # don't know if it's needed
from continuous.library import *


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

    def eval_prime(self, prime, domain):
        name = prime.derivand.string
        obs_inds = prime.derivand.indices

        #if obs_inds is None:
        #    data_arr = self.data_dict[name]
        #else:
        data_arr = self.data_dict[name][..., *obs_inds]

        data_slice = get_slice(data_arr, domain)

        orders = prime.derivative.get_spatial_orders()
        dimorders = [orders[i] for i in range(self.n_dimensions-1)]
        #dimorders = [order for _, order in prime.derivative.get_spatial_orders(max_idx=self.n_dimensions-1)]
        dimorders += prime.derivative.torder
        #if sum(dimorders) > 0:
        #    dimorders = [order for _, order in prime.derivative.get_spatial_orders()]
        return diff(data_slice, dimorders, self.dxs)
        #return data_slice
    
    def make_libraries(self, max_complexity=4, max_observables=3):
        self.libs = dict()
        terms = generate_terms_to(max_complexity, observables=self.observables,
                                  max_observables=max_observables)
        for irrep in self.irreps:
            match irrep:
                case int():
                    return full_basis(base_weight, n_dimensions, irrep)
                case FullRank():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep.rank], irrep)
                case Antisymmetric():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep.rank 
                                                    and term.symmetry() != 1], irrep)
                case SymmetricTraceFree():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep.rank 
                                                    and term.symmetry() != -1], irrep)
                #case _:
                #    raise NotImplemented

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
        for primes in term.primes:
            xorder = prime.derivative.xorder
            torder = prime.derivative.torder
            name = prime.observable.string
            if torder + xorder > 0:
                product *= scale_dict[name]['std']
            else:
                product *= scale_dict[name]['mean']
            # scale by grid spacings for derivatives (should already be accounted for by findiff)
            # product /= dx**xorder
            # product /= dt**torder
        return product if product > 0 else 1  # if the variable is always 0 then we'll get division by zero
