import numpy as np
import pennylane as qml
from jax import jit
import jax.numpy as jnp
from opt_einsum import contract
import ANNNIgen
import mpsgen
import qcnn
import os
from typing import Tuple, List, Callable
import itertools
import tqdm
import optax

import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class state:
    def __init__(self, L : int, h : float, k : float, shapes : np.ndarray, tensors : np.ndarray, towave_func : Callable):
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

        self.towave = lambda : towave_func(self.MPS)
        
class mps:
    def __init__(self, folder : str = '../tensor_data/', gpu : bool = False):
        ###########################
        #      CHECK FOLDER       #
        ###########################
        self.path = folder

        self.col3 = [[0.456, 0.902, 0.635, 1],
                     [0.400, 0.694, 0.800, 1],
                     [0.922, 0.439, 0.439, 1]]

        self.col4 = [[0.456, 0.902, 0.635, 1],
                     [0.400, 0.694, 0.800, 1],
                     [1.000, 0.514, 0.439, 1],
                     [0.643, 0.012, 0.435, 1]]
        # Check if folder exists
        try: 
            files_all        = np.array(os.listdir(folder))
            self.files_shape = files_all[np.char.startswith(files_all,'shapes_sites')]
            self.files_tens  = files_all[np.char.startswith(files_all,'tensor_sites')]
        except:
            raise TypeError(f'Folder {folder} not found')
        
        def get_info(file_string : str) -> Tuple[int, float, float, int, int]:
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
        
        # Check if files are okay
        Ls_shape, hs_shape, ks_shape, hs_shape_prec, ks_shape_prec = [], [], [], [], []
        Ls_tens,  hs_tens,  ks_tens,  hs_tens_prec,  ks_tens_prec  = [], [], [], [], []
        for file in self.files_shape:
            L, h, k, hprec, kprec = get_info(file)
            Ls_shape.append(L)
            hs_shape.append(h)
            ks_shape.append(k)
            hs_shape_prec.append(hprec)
            ks_shape_prec.append(kprec)
        for file in self.files_tens:
            L, h, k, hprec, kprec = get_info(file)
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

        # Check on h and k
        #  None for now
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
        self.MPS     = np.empty((self.n_states), dtype=object)
        self.Hparams = np.full((self.n_states,2), np.nan)
        # We save the labels as integers (0,1,2,3) and as 2-qubits states
        # for an easier implementaion with the QCNN
        self.labels3, self.labels4 = np.full(self.n_states, np.nan, dtype=int), np.full(self.n_states, np.nan, dtype=int)
        self.probs3, self.probs4   = np.full((self.n_states,4), np.nan, dtype=int), np.full((self.n_states,4), np.nan, dtype=int)
        # Well load all the states using h as inner variable loop
        # With this, I am assuming the variables h and k are disposed
        # into a grid, uniformly spaced
        # TODO: Read directly from the folder

        def toprob(label):
            prob = np.array([0]*4)
            prob[3-label] = 1
            return prob
        
        for i, (h,k) in enumerate(itertools.product(self.hs, self.ks)):
            y3, y4 = ANNNIgen.get_labels(h,-k)
            shapes  = np.loadtxt(self.shape_str(h,k)).astype(int)
            tensors = np.loadtxt(self.tensor_str(h,k))

            self.Hparams[i] = h, k
            self.labels3[i], self.labels4[i] = y3, y4
            self.probs3[i],  self.probs4[i]  = toprob(y3), toprob(y4)
            self.MPS[i] = state(self.L, h, k, shapes, tensors, self.get_psi)

        # Additionally, create a mask for the points in the axes (analitical)
        self.mask_analitical = np.logical_or(self.Hparams[:,0] == 0, self.Hparams[:,1] == 0) # type: ignore
        self.mask_analitical_ferro = np.logical_or(np.logical_and(self.Hparams[:,0] < .5, self.Hparams[:,1] == 0),
                                                   np.logical_and(self.Hparams[:,0] == 0, self.Hparams[:,1] <= 1))
        self.mask_analitical_para  = np.logical_and(self.Hparams[:,0] >= .5, self.Hparams[:,1] == 0)
        self.mask_analitical_anti  = np.logical_and(self.Hparams[:,0] ==  0, self.Hparams[:,1] >  1)
        
        self.qcnn = qcnn.qcnn(self.L)

    def train(self, epochs, PSI, Y, opt_state, show_progress = False):
        params = self.qcnn.PARAMS

        progress = tqdm.tqdm(range(epochs), position=0, leave=True)
        for epoch in range(epochs):
            params, opt_state, train_loss, accuracy = self.qcnn.update(opt_state, PSI, params, Y)

            # Update progress bar
            progress.update(1)
            progress.set_description(f'Loss: {train_loss:.5f}')
            
        self.qcnn.PARAMS = params 
        return opt_state
    
    def train3(self, epochs = 100, train_indices = [], batch_size = 0, lr = 1e-2):
        self.qcnn.optimizer = optax.adam(learning_rate=lr)

        if len(train_indices) == 0:
            # Set the analytical points as training inputs
            train_indices = np.arange(len(self.MPS)).astype(int)[self.mask_analitical]

        opt_state = self.qcnn.optimizer.init(self.qcnn.PARAMS)
        if batch_size == 0:
            STATES = [mpsclass.towave() for mpsclass in self.MPS[train_indices]]
            STATES = jnp.array(STATES)
            YSTATES      = self.probs3[train_indices]
            Y      = self.labels3[train_indices]
            self.train(epochs, STATES, YSTATES, opt_state)

    def predict(self, batch_size = 0, plot = False, eachclass = False):
        if batch_size == 0:
            STATES = [mpsclass.towave() for mpsclass in self.MPS]
            STATES = jnp.array(STATES)
            
            PREDICTIONS = self.qcnn.jv_q_circuit(STATES, self.qcnn.PARAMS)


        ARGPREDICTIONS = np.argmax(PREDICTIONS, axis=1)
        print(np.shape(PREDICTIONS))
    
        if plot: 
            ANNNIgen.plot_layout(self, False, True, True, 'prediction', figure_already_defined = False)
            plt.imshow(np.flip(np.reshape(ARGPREDICTIONS, (len(self.hs), len(self.ks))), axis=0))

            if eachclass:
                fig = plt.figure(figsize=(24,5))
                for k in range(4):
                    fig.add_subplot(1,4,k+1)
                    plt.title(f'Class {k}')
                    ANNNIgen.plot_layout(self, True, True, True, title='', figure_already_defined = True)
                    plt.imshow(np.flip(np.reshape(PREDICTIONS[:,k], (len(self.hs), len(self.ks))), axis=0))

    def plot_labels(self):
        cm3 = ListedColormap(self.col3, name='color3')
        cm4 = ListedColormap(self.col4, name='color4')
        fig = plt.figure(figsize=(15,12))
        ax1 = fig.add_subplot(1,2,1)
        ANNNIgen.plot_layout(self, True, True, True, '3 Phases', figure_already_defined = True)
        ax1.imshow(np.rot90(np.reshape(self.labels3, (len(self.hs), len(self.ks)))), cmap=cm3)
        ax2 = fig.add_subplot(1,2,2)
        ANNNIgen.plot_layout(self, True, True, True, '3 Phases + floating phase', figure_already_defined = True)
        ax2.imshow(np.rot90(np.reshape(self.labels4, (len(self.hs), len(self.ks)))), cmap=cm4)
        