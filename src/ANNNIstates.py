# General
from jax import jit
import jax.numpy as jnp
import numpy as np

from opt_einsum import contract
# Quantum
import optax
import pennylane as qml

# Plot
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

# Utils
import os
from typing import Tuple, List, Callable
from numpy.typing import NDArray
import itertools
import tqdm

# Custom
import ANNNIgen
import mpsgen
import qcnn

class state:
    def __init__(self, L : int, h : float, k : float, shapes : NDArray, tensors : NDArray, _towave_func : Callable):
        """Single ANNNI MPS class

        Class containing a single MPS object for the ANNNI model

        Parameters
        ----------
        L : int 
            Number of sites
        h : float
            Value of h
        k : float or NDArray
            Value of k
        shapes : NDArray
            Matrix containing the shape of each site
        tensors : NDArray
            Simple 1D list of the tensor values
        _towave_func : Callable
            Funciton to transform the MPS into a wavefunction
            (INTERNAL FUNCTION TO THE CLASS MPS)
        """
        # Store the input values:
        self.L, self.h, self.k = L, h, k
        self.shapes = shapes.astype(int)
        # This needs some explanation:
        # 1. Start from `tensors`, it is just a list of value that have to be grouped in smaller
        #    tensors according to the shapes list, that defines the list of the smaller tensors
        # 2. Compute self.splits, it tells the indexes for splitting the full vector into
        #    the smaller tensors
        #    EX:
        #        [a,b,c,d,e,f,g,h] : full vector
        #         0 1 2 3 4 5 6 7    index
        #        [2,6]             : self.splits
        #        self.splits tells you how to break the full vector
        #        [a,b] [c,d,e,f] [g,h] : split vectors
        self.splits = np.cumsum(np.prod(shapes, axis=1)).astype(int)
        # 3. Compute the MPS:
        #     3.1: Group the elements in split vectors (not split tensors yet)
        #          np.array_split(tensors, self.splits)[:-1] 
        #          flat_tn = np.array_split(tensors, self.splits)[:-1] split vectors, List of L vectors
        #                    NOTE: np.array_split(tensors, self.splits)[-1] is empty
        #     3.2: Reshape each of the L vector into tensor with the shape declared in self.shape
        # 3.1 and 3.2 done together to avoid garbage collector shenanigans
        self.MPS = [site.reshape(self.shapes[i]) for i, site in enumerate(np.array_split(tensors, self.splits)[:-1])]

        self.towave = lambda : _towave_func(self.MPS)
        
class mps:
    def __init__(self, folder : str = '../tensor_data/', gpu : bool = False):
        """ANNNI MPS class

        Main class for the data analysis, it is a continer of: 
        > Multiple MPS
        > QCNN
        > QCNN training functions
        > Plot functions

        Parameters
        ----------
        folder : str
            Folder from where to read all necessary data
        gpu : bool
            if True, it tries to use the GPU with jax backend
        """

        def _get_info(file_string : str) -> Tuple[int, float, float, int, int]:
            """
            1. Split big string into array of string
            'shapes_sites_ANNNI_L_{L}_h_{h}_kappa_{k}'
                into
            ['shapes', 'sites', 'ANNNI', 'L', '{L}','h', '{h}', 'kappa', '{k}']
              0         1        2        3    4^    5    6^     7        8^
            2. Take the element 4, 6 and 8 
            """
            split_str = file_string.split('_')

            # return respectively:
            # L, h, k, precision on h, precision on k
            return int(split_str[4]), float(split_str[6]), float(split_str[8]), len(split_str[6].split('.')[1]), len(split_str[8].split('.')[1])
        
        def _toprob(label):
            """
            Converts a label into a probability vector of a 2-qubits state. 
            It is needed in the loss function of the QCNN since it uses the 
            Multi-class cross entropy 

            * 0 -> [1,0,0,0]
            * 1 -> [0,1,0,0]
            * 2 -> [0,0,1,0]
            * 3 -> [0,0,0,1]
            """
            prob = np.array([0]*4)
            prob[label] = 1
            return prob
        
        ###########################
        #     VALUES ASSIGNS.     #
        ###########################
        self.path = folder

        # Color for representing plots with
        #   3 Phases (Ferromagnetic, Paramagnetic, Antiphase)
        #   4 Phases : 3 Phases + Floating Phase
        self.col3 = [[0.456, 0.902, 0.635, 1],
                     [0.400, 0.694, 0.800, 1],
                     [0.922, 0.439, 0.439, 1]]

        self.col4 = [[0.456, 0.902, 0.635, 1],
                     [0.400, 0.694, 0.800, 1],
                     [1.000, 0.514, 0.439, 1],
                     [0.643, 0.012, 0.435, 1]]
        # Relative colormaps
        self.cm3 = ListedColormap(self.col3, name='color3')
        self.cm4 = ListedColormap(self.col4, name='color4')

        ###########################
        #      CHECK FOLDER       #
        ###########################
        
        # Check if folder exists
        try: 
            files_all        = np.array(os.listdir(folder))
            self.files_shape = files_all[np.char.startswith(files_all,'shapes_sites')]
            self.files_tens  = files_all[np.char.startswith(files_all,'tensor_sites')]
        except:
            raise TypeError(f'Folder {folder} not found')
        
        # Check if files are okay
        Ls_shape, hs_shape, ks_shape, hs_shape_prec, ks_shape_prec = [], [], [], [], []
        Ls_tens,  hs_tens,  ks_tens,  hs_tens_prec,  ks_tens_prec  = [], [], [], [], []
        for file in self.files_shape:
            L, h, k, hprec, kprec = _get_info(file)
            Ls_shape.append(L)
            hs_shape.append(h)
            ks_shape.append(k)
            hs_shape_prec.append(hprec)
            ks_shape_prec.append(kprec)
        for file in self.files_tens:
            L, h, k, hprec, kprec = _get_info(file)
            Ls_tens.append(L)
            hs_tens.append(h)
            ks_tens.append(k)
            hs_tens_prec.append(hprec)
            ks_tens_prec.append(kprec)

        # Check on L
        if len(np.unique(Ls_shape)) > 1 or len(np.unique(Ls_tens)) > 1:
            raise ValueError(f'L has multiple values')
        elif Ls_shape[0] != Ls_tens[0]:
            raise ValueError(f'L has inconsistent values')
        # otherwise L is okay:
        self.L = Ls_shape[0]

        # TODO: Check on h and k

        self.hs = np.sort(np.unique(hs_shape))
        self.ks = np.sort(np.unique(ks_shape))

        # Check on precisions
        if len(np.unique(hs_shape_prec + hs_tens_prec)) > 1 or len(np.unique(ks_shape_prec + ks_tens_prec)) > 1: 
            raise ValueError('Inconsistent precisions in files')
        self.h_prec = hs_shape_prec[0]
        self.k_prec = ks_shape_prec[0]

        # Format of the file names:
        # shape_file  : shape_sites_ANNNI_L_{N}_h_{h}_kappa_{k}
        self.shape_str  = lambda h, k : folder+f'shapes_sites_ANNNI_L_{self.L}_h_{h:.{self.h_prec}f}_kappa_{k:.{self.k_prec}f}'
        # tensor_file : shape_sites_ANNNI_L_{N}_h_{h}_kappa_{k}
        self.tensor_str = lambda h, k : folder+f'tensor_sites_ANNNI_L_{self.L}_h_{h:.{self.h_prec}f}_kappa_{k:.{self.k_prec}f}'

        ###########################
        #      LOAD ALL MPS       #
        ###########################
        self.mpstowavefunc_subscript = mpsgen.get_subscript(self.L)
        if gpu:
            self.get_psi = lambda TT: contract(self.mpstowavefunc_subscript, *TT, backend='jax').flatten()
        else:
            self.get_psi = lambda TT: contract(self.mpstowavefunc_subscript, *TT).flatten() # type: ignore

        self.n_states = len(self.hs)*len(self.ks)
        self.MPS      = np.empty((self.n_states), dtype=object)
        self.Hparams  = np.full((self.n_states,2), np.nan)
        # We save the labels as integers (0,1,2,3) and as 2-qubits states
        # for an easier implementaion with the QCNN
        self.labels3, self.labels4 = np.full(self.n_states, np.nan, dtype=int), np.full(self.n_states, np.nan, dtype=int)
        self.probs3, self.probs4   = np.full((self.n_states,4), np.nan, dtype=int), np.full((self.n_states,4), np.nan, dtype=int)
        # Well load all the states using h as inner variable loop
        # With this, I am assuming the variables h and k are disposed
        # into a grid, uniformly spaced
        # TODO: Read directly from the folder
        
        for i, (h,k) in enumerate(itertools.product(self.hs, self.ks)):
            # Weird for loop, I know, it is equivalent of performing a nested for loops for h in hs and k in ks
            # + the enumerate loop variable:
            # i | h | k |
            # ----------|
            # 0 | 0 | 0 |
            # 1 | 1 | 0 | 
            # ...
            y3, y4 = ANNNIgen.get_labels(h,k)
            shapes  = np.loadtxt(self.shape_str(h,k)).astype(int)
            tensors = np.loadtxt(self.tensor_str(h,k))

            self.Hparams[i] = h, k
            self.labels3[i], self.labels4[i] = y3, y4
            self.probs3[i],  self.probs4[i]  = _toprob(y3), _toprob(y4)
            self.MPS[i] = state(self.L, h, k, shapes, tensors, self.get_psi)

        # In the next comments, by analytical I refer to the points
        # being in either of the two axes (k = 0 OR h = 0 (or being inclusive))
        # those are the points which labels can be obtained through
        # analytical computation 

        # Additionally, create a mask for all the analytical points
        self.mask_analitical = np.logical_or(self.Hparams[:,0] == 0, self.Hparams[:,1] == 0) # type: ignore

        # Mask for all the analytical ferromagnetic points
        self.mask_analitical_ferro = np.logical_or(np.logical_and(self.Hparams[:,0] <  1, self.Hparams[:,1] == 0),
                                                   np.logical_and(self.Hparams[:,0] == 0, self.Hparams[:,1] <= .5))
        # Mask for all the analytical paramagnetic points
        self.mask_analitical_para  = np.logical_and(self.Hparams[:,0] >= 1, self.Hparams[:,1] == 0)
        # Mask for all the analytical antiphase points
        self.mask_analitical_anti  = np.logical_and(self.Hparams[:,0] ==  0, self.Hparams[:,1] >  .5)
        
        self.qcnn = qcnn.qcnn(self.L)

    def _train(self, epochs, PSI, Y, opt_state):
        """
        Internal simple training function, to be called by other functions
        """
        params = self.qcnn.PARAMS # Get the parameters from the class

        progress = tqdm.tqdm(range(epochs), position=0, leave=True)
        for epoch in range(epochs):
            # TODO: Fix accuracy function
            params, opt_state, train_loss, accuracy = self.qcnn.update(opt_state, PSI, params, Y)

            # Update progress bar
            progress.update(1)
            progress.set_description(f'Loss: {train_loss:.5f}')
            
        # Update the parameters after the training and return the new optimizer
        # state, (for performing multiple trainings)
        self.qcnn.PARAMS = params 
        return opt_state
    
    # TODO: Add batching functionality
    def train3(self, epochs : int = 100, train_indices : NDArray = np.array([]), 
                     batch_size : int = 0, lr : float = 1e-2):
        """Training function for 3 classes

        Training function assuming 3 Phases: 
        Ferromagnetic, Paramagnetic and Antiphase

        Parameters
        ----------
        epochs : int
            Number of epochs
        train_indices : NDArray
            Indexes of points to train
        batch_size : int
            Number of MPS to used as input in a batch
        lr : float
            Learning Rate
        """
        self.qcnn.optimizer = optax.adam(learning_rate=lr)

        if len(train_indices) == 0:
            # Set the analytical points as training inputs
            train_indices = np.arange(len(self.MPS)).astype(int)[self.mask_analitical]
        else:         
            train_indices = train_indices

        opt_state = self.qcnn.optimizer.init(self.qcnn.PARAMS)
        if batch_size == 0:
            STATES  = jnp.array([mpsclass.towave() for mpsclass in self.MPS[train_indices]])
            YPROBS = self.probs3[train_indices]
            # Y     = self.labels3[train_indices]
            self._train(epochs, STATES, YPROBS, opt_state)
        else: 
            raise NotImplementedError("TODO: Batching")

    # TODO: Add batching functionality
    def predict(self, batch_size : int = 0, plot : bool = False, eachclass : bool = False):
        """Output the predicted phases

        Output the predicted phases

        Parameters
        ----------
        batch_size : int
            Number of MPS to used as input in a batch
        plot : bool
            if True, it plots
        eachclass : bool 
            if True, it plots the probability of each class separately
        """
        if batch_size == 0:
            STATES = jnp.array([mpsclass.towave() for mpsclass in self.MPS])
            
            PREDICTIONS = self.qcnn.jv_q_circuit(STATES, self.qcnn.PARAMS)
        else: 
            raise NotImplementedError("TODO: Batching")

        # PREDICTIONS is a (h*k, 4) array
        # The values in the axis=1 represent the probability of each class. 
        # Computing the argmax on that axis outputs the class the model is most
        # confident of
        ARGPREDICTIONS = np.argmax(PREDICTIONS, axis=1)

        cmap = self.cm3
        # If it detected a 4th class, the colormap is different
        if len(np.unique(ARGPREDICTIONS)) == 4:
            cmap = self.cm4 
    
        if plot: 
            ANNNIgen.plot_layout(self, True, True, True, 'prediction', figure_already_defined = False)
            plt.imshow(np.flip(np.reshape(ARGPREDICTIONS, (len(self.hs), len(self.ks))), axis=0), cmap=cmap)

            if eachclass:
                fig = plt.figure(figsize=(20,6))
                for k in range(4):
                    haxis = False if k > 0 else True
                    fig.add_subplot(1,4,k+1)
                    plt.title(f'Class {k}')
                    ANNNIgen.plot_layout(self, True, True, True, title='', haxis=haxis,  figure_already_defined = True)
                    im = plt.imshow(np.flip(np.reshape(PREDICTIONS[:,k], (len(self.hs), len(self.ks))), axis=0), vmin = 0, vmax = 1)
                
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.82, 0.25, 0.02, 0.5]) # type: ignore
                fig.colorbar(im, cax=cbar_ax)

    def plot_labels(self):
        """Plot the 'true' phases (the labels) of the ANNNI model
        """
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(1,2,1)
        plt.title('Labels (3)')
        ANNNIgen.plot_layout(self, True, True, True, '3 Phases', figure_already_defined = True)
        ax1.imshow(np.flip(np.reshape(self.labels3, (len(self.hs), len(self.ks))), axis=0), cmap=self.cm3)
        ax2 = fig.add_subplot(1,2,2)
        plt.title('Labels (4)')
        ANNNIgen.plot_layout(self, True, True, True, '3 Phases + floating phase', haxis = False, figure_already_defined = True)
        ax2.imshow(np.flip(np.reshape(self.labels4, (len(self.hs), len(self.ks))), axis=0), cmap=self.cm4)
        