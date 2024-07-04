import tenpy
import tenpy.networks.site as tns
from tenpy.models.model import MPOModel, CouplingModel
from tenpy.networks.mps import MPS
import numpy as np
from tenpy.algorithms import dmrg
import os 
import tqdm
import argparse

parser = argparse.ArgumentParser(description='DMRG states of the ANNNI model')

parser.add_argument('--side', type=int, default=51, metavar='INT', 
                    help='Discretization of h and k')

parser.add_argument('--L', type=int, default=12, metavar='INT', 
                    help='Number of spins')

parser.add_argument('--chi', type=int, default=12, metavar='INT', 
                    help='Maximum bond dimension')

parser.add_argument(
    "--hide",
    action="store_true",
    help="suppress progress bars",
)

args = parser.parse_args()

class ANNNI(CouplingModel):
    def __init__(self, model_param):
        self.lat = self.init_lattice(model_param)
        self.init_terms(model_param)
        self.name = 'ANNNI'
        self.dtype = np.double


    def init_sites(self, model_params):
        conserve = model_params.get('conserve', None)
        #self.logger.info("%s: set conserve to %s", self.name, conserve)
        site = tns.SpinHalfSite(conserve)
        return [site]


    def init_lattice(self, model_params):
        L = model_params.get("L", 2)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'periodic' if bc_MPS == 'infinite' else 'open'
        sites = self.init_sites(model_params)
        lat = tenpy.models.lattice.Lattice([L], sites, bc=bc, bc_MPS=bc_MPS)
        return lat


    def init_terms(self, model_params):
        CouplingModel.__init__(self, self.lat)
        j = model_params.get("j", 1.0)
        h = model_params.get("h", 1.0)
        k = model_params.get("k", 0.0)
        #lamb = model_params.get("lamb", 0.0)
        self.explicit_plus_hc =False
        self.add_multi_coupling(-j, [('Sigmax',0,0),('Sigmax',1,0)],plus_hc=False)
        self.add_multi_coupling(+k, [('Sigmax',0,0),('Sigmax',2,0)],plus_hc=False)
        self.add_onsite(h, 0, 'Sigmaz')
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())

    def save_sites(self, tensors, path, model_params, precision=3):

        L = model_params.get("L", 2)
        h = model_params.get("h", 1.0)
        kappa = model_params.get("k", 0.0)
        shapes = tensor_shapes(tensors)
        np.savetxt(
            f"{path}/shapes_sites_{self.name}_L_{L}_h_{h:.{precision}f}_kappa_{kappa:.{precision}f}",
            shapes,
            fmt="%1.i",
        )

        tensor = [element for site in tensors for element in site.flatten()]
        np.savetxt(
            f"{path}/tensor_sites_{self.name}_L_{L}_h_{h:.{precision}f}_kappa_{kappa:.{precision}f}",
            tensor,
        )

    def load_sites(self, path, model_params, precision=3):

            L = model_params.get("L", 2)
            h = model_params.get("h", 1.0)
            kappa = model_params.get("k", 0.0)
            shapes = np.loadtxt(
                f"{path}shapes_sites_{self.name}_L_{L}_h_{h:.{precision}f}_kappa_{kappa:.{precision}f}"
            ).astype(int)

            filedata = np.loadtxt(
                f"{path}tensor_sites_{self.name}_L_{L}_h_{h:.{precision}f}_kappa_{kappa:.{precision}f}"
            )

            labels = tns.get_labels(shapes)
            flat_tn = np.array_split(filedata, labels)
            flat_tn.pop(-1)

            tensors = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]
            return tensors
    
def tensor_shapes(lists):
    shapes = [array.shape for array in lists]
    return shapes

# Computing simulations
L = args.L
chi = args.chi

initial_state = ['up'] * L
path = f'./L{L}_X{args.chi}'
hs = np.linspace(0.0,2.0,args.side); ks = np.linspace(0.0,1.0,args.side)

master_path = './'
L_path = f'{master_path}{L}/'
X_path = f'{L_path}{chi}/'

for path in [master_path, L_path, X_path]:
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

progress = tqdm.tqdm(range(args.side*args.side), disable=args.hide)
for h in hs:
        for k in ks:
                progress.update(1)
                open_data = dict()
                model_params = dict(L = L, J = 1.0, k = k, h = h , bc_MPS = 'finite')
                bc_MPS = model_params["bc_MPS"]
                model = ANNNI(model_params)
                psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc = bc_MPS)
                dmrg_params = {'mixer': False,
                        'chi_list': {0:chi,
                                # 8:60,
                                },
                        # 'max_E_err': 1.e-7,
                        # 'max_S_err': 1.e-3,
                        'min_sweeps': 10,
                        'max_sweeps': 4,
                        'N_sweeps_check': 4,
                        'combine': True,
                        'norm_tol' : 1,
                        }
                eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
                E, psi = eng.run()
                open_data['E'] = E
                S = psi.entanglement_entropy()
                open_data['S'] = S
                # correlation functions
                X = psi.expectation_value("Sigmax")
                Z = psi.expectation_value("Sigmaz")
                open_data['X'] = X
                open_data['Z'] = Z
                i0 = psi.L // 4  # for fixed `i`
                j = np.arange(i0 + 1, psi.L)
                XX = psi.term_correlation_function_right([("Sigmax", 0)], [("Sigmax", 0)], i_L=i0, j_R=j)
                XX_disc = XX - X[i0] * X[j]
                ZZ = psi.term_correlation_function_right([("Sigmaz", 0)], [("Sigmaz", 0)], i_L=i0, j_R=j)
                ZZ_disc = ZZ - Z[i0] * Z[j]
                open_data['XX'] = XX_disc
                open_data['ZZ'] = ZZ_disc

                # filename = f"{path}open_properties_L_{L}_h_{h:.{1}f}_kappa_{k:.{3}f}.pickle"
                # with open(filename, "wb") as file:
                #        pickle.dump(open_data, file)

                tensors = [psi.get_B(i)._data[0] for i in range(L)]
                model.save_sites(tensors, path, model_params)
