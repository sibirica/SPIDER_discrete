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
        self.pad = 0
        #return domains

    def eval_prime(self, prime, domain):
        name = prime.derivand.string
        obs_inds = map(lambda idx: idx.value, prime.derivand.indices) # unpack the indices

        #if obs_inds is None:
        #    data_arr = self.data_dict[name]
        #else:
        data_arr = self.data_dict[name][..., *obs_inds]

        data_slice = get_slice(data_arr, domain)

        orders = prime.derivative.get_spatial_orders()
        dimorders = [orders[i] for i in range(self.n_dimensions-1)]
        #dimorders = [order for _, order in prime.derivative.get_spatial_orders(max_idx=self.n_dimensions-1)]
        dimorders += [prime.derivative.torder]
        #if sum(dimorders) > 0:
        #    dimorders = [order for _, order in prime.derivative.get_spatial_orders()]
        return diff(data_slice, dimorders, self.dxs) if sum(dimorders)>0 else data_slice
        #return data_slice
    
    def make_libraries(self, max_complexity=4, max_observables=3):
        self.libs = dict()
        terms = generate_terms_to(max_complexity, observables=self.observables,
                                  max_observables=max_observables)
        for irrep in self.irreps:
            match irrep:
                case int():
                    self.libs[irrep] = LibraryData([term for term in terms if term.rank == irrep], irrep)
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

    def find_scales(self, names=None):
        # find mean/std deviation of fields in data_dict that are in names
        self.scale_dict = dict()
        for name in self.data_dict:
            if names is None or name in names:
                self.scale_dict[name] = dict()
                # if these are vector quantities the results could be wonky in the unlikely
                # case a vector field is consistently aligned with one of the axes
                self.scale_dict[name]['mean'] = np.mean(
                    np.linalg.norm(self.data_dict[name]) / np.sqrt(self.data_dict[name].size))
                self.scale_dict[name]['std'] = np.std(self.data_dict[name])

    def get_char_size(self, term):
        # return characteristic size of a library term
        product = 1
        for prime in term.primes:
            xorder = prime.derivative.xorder
            torder = prime.derivative.torder
            name = prime.derivand.string
            if torder + xorder > 0:
                product *= self.scale_dict[name]['std']
            else:
                product *= self.scale_dict[name]['mean']
            product /= self.xscale ** xorder
            product /= self.tscale ** torder
        #print(f'char size of {term} is {product}')
        return product if product > 0 else 1 # if the variable is always 0 then we'll get division by zero
