import numpy as np
import os 
import tqdm
import argparse
from ANNNI.hamiltonian import Hamiltonian 

parser = argparse.ArgumentParser(description='Exact states of the ANNNI model')

parser.add_argument('--side', type=int, default=51, metavar='INT', 
                    help='Discretization of h and k')

parser.add_argument('--L', type=int, default=12, metavar='INT', 
                    help='Number of spins')

parser.add_argument(
    "--hide",
    action="store_true",
    help="suppress progress bars",
)

args = parser.parse_args()

# Computing simulations
L = args.L

hs = np.linspace(0.0,2.0,args.side); ks = np.linspace(0.0,1.0,args.side)

master_path = './'
path = f'{master_path}{L}/diag/'

if not os.path.exists(f'{master_path}{L}'):
    os.makedirs(f'{master_path}{L}')
if not os.path.exists(f'{path}'):
    os.makedirs(f'{path}')

Ham = Hamiltonian(L, np.vstack((hs,ks)).T)

progress = tqdm.tqdm(range(args.side*args.side), disable=args.hide)
for h in hs:
    for k in ks:
        E_filename = f"E_L_{L}_h_{h:.{3}f}_kappa_{k:.{3}f}"
        psi_filename = f"psi_L_{L}_h_{h:.{3}f}_kappa_{k:.{3}f}"
        progress.update(1)
        if (E_filename not in os.listdir(path)) and (psi_filename not in os.listdir(path)):
            E, psi = Ham.get_values((args.L, h, k))
            np.save(f'{path}{E_filename}', E)
            np.save(f'{path}{psi_filename}', psi)

        else:
            print(f'MPS: L:{L}, h:{h}, k:{k} already exists')