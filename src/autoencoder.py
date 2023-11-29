""" This module implements the base functions to implement an anomaly detector model"""
import pennylane as qml
from pennylane import numpy as np
from jax import value_and_grad, jit, vmap, numpy as jnp
import optax

import circuits
from typing import List, Callable
from numbers import Number

##############


def encoder_circuit(N: int, state, params: List[Number]) -> int:
    """
    Building function for the circuit Encoder(params)

    Parameters
    ----------
    N : int
        Number of qubits
    params: np.ndarray
        Array of parameters/rotation for the circuit

    Returns
    -------
    int
        Number of parameters of the circuit
    """
    active_wires = np.arange(N)

    # Number of wires that will not be measured |phi>
    n_wires = N // 2 + N % 2

    wires = np.concatenate(
        (np.arange(0, n_wires // 2 + n_wires % 2), np.arange(N - n_wires // 2, N))
    )
    wires_trash = np.setdiff1d(active_wires, wires)

    qml.QubitStateVector(state, wires=range(N))
    # Visual Separation PSI||Anomaly
    qml.Barrier()
    qml.Barrier()

    index = circuits.encoder_circuit(wires, wires_trash, active_wires, params)

    return index


class encoder:
    def __init__(self, L, lr = 1e-2):
        self.n_qubits = L
        self.encoder_circuit_fun = lambda state, enc_p: encoder_circuit(L, state, enc_p)
        # Create a dummy state to compute the number of parameters needed.
        dummystate = np.zeros(2**self.n_qubits) 
        dummystate[0] = 1  # State needs to be normal

        self.n_params = self.encoder_circuit_fun(dummystate, [0] * 10000)
        self.PARAMS = np.array(np.random.rand(self.n_params))
        
        self.device = qml.device('default.qubit.jax', wires=self.n_qubits)

        self.n_wires = self.n_qubits // 2 + self.n_qubits % 2
        self.n_trash = self.n_qubits // 2
        self.wires = np.concatenate(
            (
                np.arange(0, self.n_wires // 2 + self.n_wires % 2),
                np.arange(self.n_qubits - self.n_wires // 2, self.n_qubits),
            )
        )
        self.wires_trash = np.setdiff1d(np.arange(self.n_qubits), self.wires)

        def circuit(x, p):
            self.encoder_circuit_fun(x, p)
            return [qml.expval(qml.PauliZ(int(k))) for k in self.wires_trash]

        self.q_circuit    = qml.QNode(circuit, self.device)
        self.v_q_circuit  = vmap(self.q_circuit, (0, None))
        self.jv_q_circuit = jit(self.v_q_circuit)

        def loss_fun(x, params, qcirc):
            out = jnp.mean(jnp.array(qcirc(x, params)))

            return 1 - out
                
        self.get_loss    = lambda x, p: loss_fun(x, p, self.q_circuit)
        self.v_get_loss  = vmap(self.get_loss, (0, None))
        self.jv_get_loss = jit(self.v_get_loss)

        def mean_loss(p, X):
            # self.jv_get_loss outputs the single loss X[i]<->Y[i]
            # we need to reduce the matrix to a single scalar value
            return jnp.mean(self.jv_get_loss(X, p))

        def optimizer_update(opt, opt_state, X, p):
            loss, grads = value_and_grad(mean_loss)(p, X)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state, loss
        
        # Update the optimizer
        self.optimizer = optax.adam(learning_rate=lr)
        self.update = jit(lambda opt_state, X, p: optimizer_update(self.optimizer, opt_state, X, p))

    def save_params(self, file : str):
        np.savetxt(file, self.PARAMS)

    def load_params(self, file : str):
        self.PARAMS = np.loadtxt(file)

    def __repr__(self):
        @qml.qnode(self.device, interface="jax")
        def circuit_drawer(self):
            self.encoder_circuit_fun(np.arange(self.n_params))

            return [qml.expval(qml.PauliZ(int(k))) for k in self.wires_trash]

        return qml.draw(circuit_drawer)(self)

 