import pennylane as qml
from jax import vmap, jit 
import jax.numpy as jnp
import numpy as np 
import scipy 
import tqdm

def get_H(
    N: int, L: float, K: float, ring: bool = False
) -> qml.ops.qubit.hamiltonian.Hamiltonian:
    """
    Set up Hamiltonian:
            H = J1* (- Σsigma^i_x*sigma_x^{i+1} - (h/J1) * Σsigma^i_z - (J2/J1) * Σsigma^i_x*sigma_x^{i+2} )
        
        [where J1 = 1, (h/J1) = Lambda(/L), (J2/J1) = K]

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    L : float
        h/J1 parameter
    K : float
        J1/J2 parameter
    ring : bool
        If False, system has open-boundaries condition

    Returns
    -------
    pennylane.ops.qubit.hamiltonian.Hamiltonian
        Hamiltonian Pennylane class for the (Transverse) Ising Chain
    """
    # Interaction of spins with magnetic field
    H = -L * qml.PauliZ(0)
    for i in range(1, N):
        H = H - L * qml.PauliZ(i)

    # Interaction between spins (neighbouring):
    for i in range(0, N - 1):
        H = H + (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))

    # Interaction between spins (next-neighbouring):
    for i in range(0, N - 2):
        H = H + (-1) * K * (qml.PauliX(i) @ qml.PauliX(i + 2))

    # If ring == True, the 'chain' needs to be closed
    if ring:
        # Nearest interaction between last spin and first spin -particles
        H = H + (-1) * (qml.PauliX(N - 1) @ qml.PauliX(0))
        # Next nearest interactions:
        H = H + (-1) * K * (qml.PauliX(N - 2) @ qml.PauliX(0))
        H = H + (-1) * K * (qml.PauliX(N - 1) @ qml.PauliX(1))

    return H

class Hamiltonian:
    def __init__(self, N, Hparams):
        self.params = np.insert(Hparams, 0, N, axis = 1)

    def get_values(self,params):
        Hmat = get_H(int(params[0]), params[1], params[2]).sparse_matrix().astype(np.float32)
        eigvalues, eigvectors = scipy.sparse.linalg.eigsh(Hmat)
        return eigvalues[0], eigvectors[0]

    def get_all(self):
        progress = tqdm.tqdm(range(len(self.params)))
        eigvs = []
        for params in self.params:
            eigvs.append((self.get_values(params))) 
            progress.update(1)

        return eigvs 