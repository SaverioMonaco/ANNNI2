import numpy as np 
import pennylane as qml 
import circuits 
from jax import vmap, jit, numpy as jnp, value_and_grad 
import optax

class qcnn:
    def __init__(self, L, PARAMS = [], lr = 1e-2):
        """
        This class requires:
            STATES: an array of L wavefunctions (L, 2**n_qubits)
            PARAMS: an array of the parameters of the ANNNI model relative to the STATES wavefunction
                    PARAMS = (L, 2)
                                 |
                                h and kappa
        """ 

        # number of initial qubits of the QCNN: 
        self.n_qubits  = L
        self.n_outputs = 2

        self.PARAMS = PARAMS 

        def qcnn_circuit(state, params):
            # Wires that are not measured (through pooling)
            active_wires = np.arange(self.n_qubits)

            qml.QubitStateVector(state, wires=range(self.n_qubits))

            # Visual Separation State||QCNN
            qml.Barrier()
            qml.Barrier()

            # Index of the parameter vector
            index = 0

            # Iterate Convolution+Pooling until we only have a single wires
            index = circuits.wall_gate(active_wires, qml.RY, params, index)
            circuits.wall_cgate_serial(active_wires, qml.CNOT)
            while len(active_wires) > self.n_outputs:  # Repeat until the number of active wires
                # (non measured) is equal to n_outputs
                # Convolute
                index = circuits.convolution(active_wires, params, index)
                # Measure wires and apply rotations based on the measurement
                index, active_wires = circuits.pooling(active_wires, qml.RX, params, index)

                qml.Barrier()

            circuits.wall_cgate_serial(active_wires, qml.CNOT)
            index = circuits.wall_gate(active_wires, qml.RY, params, index)

            # Return the number of parameters
            return index + 1, active_wires

        self.device = qml.device('default.qubit.jax', wires=self.n_qubits)
        
        self.circuit_fun  = qcnn_circuit

        # Create a dummy state to compute the number of parameters needed.
        dummystate = np.zeros(2**self.n_qubits) 
        dummystate[0] = 1  # State needs to be normal
        self.n_params, self.final_active_wires = self.circuit_fun(dummystate, np.zeros(10000)) 
        if len(self.PARAMS) == 0: # if the parameters are the default ones...
            self.PARAMS = np.random.normal(loc = 0, scale = 1, size = (self.n_params, )) # mean  = 0
                                                                                         # stdev = 1 roghly each point is
                                                                                         # in [-pi, +pi]

        def circuit(x, p):
            self.circuit_fun(x, p)
            return qml.probs([int(k) for k in self.final_active_wires])

        self.q_circuit    = qml.QNode(circuit, self.device)
        self.v_q_circuit  = vmap(self.q_circuit, (0, None))
        self.jv_q_circuit = jit(self.v_q_circuit)

        def loss_fun(x, params, y, qcirc):
            out = qcirc(x, params)
            logprob = jnp.log(out)

            return -jnp.multiply(logprob, y)
                
        self.get_loss    = lambda x, p, y: loss_fun(x, p, y, self.q_circuit)
        self.v_get_loss  = vmap(self.get_loss, (0, None, 0))
        self.jv_get_loss = jit(self.v_get_loss)

        def mean_loss(p, X, Y):
            # self.jv_get_loss outputs the single loss X[i]<->Y[i]
            # we need to reduce the matrix to a single scalar value
            return jnp.mean(self.jv_get_loss(X, p, Y))

        def optimizer_update(opt, opt_state, X, p, Y):
            loss, grads = value_and_grad(mean_loss)(p, X, Y)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state, loss, 0
        
        # Update the optimizer
        self.optimizer = optax.adam(learning_rate=lr)
        self.update = jit(lambda opt_state, X, p, Y: optimizer_update(self.optimizer, opt_state, X, p, Y))

    def reset_params(self, loc = 0, scale = 1):
        """Reset the parameters of the QCNN

        Reinitialize the parameters of the QCNN according to a gaussian distribution

        Parameters
        ----------
        loc : float 
            Mean of the gaussian distribution
        scale : float
            Deviation of the gaussian distribution
        """
        self.PARAMS = np.random.normal(loc = loc, scale = scale, size = (self.n_params, )) # mean  = 0
                                                                                           # stdev = 1 roghly each point is
                                                                                           # in [-pi, +pi]

    def save_params(self, file : str):
        np.savetxt(file, self.PARAMS)

    def load_params(self, file : str):
        self.PARAMS = np.loadtxt(file)

    
    def __repr__(self):
        # Create a dummy state to compute the number of parameters needed.
        dummystate = np.zeros(2**self.n_qubits) 
        dummystate[0] = 1  # State needs to be normal

        reprstr  = f'Number of qubits : {self.n_qubits}\n'
        reprstr += f'Number of params : {self.n_params}\n\n'
        reprstr += 'Circuit:\n'
        reprstr += qml.draw(self.q_circuit)(dummystate, np.arange(self.n_params))
        return reprstr