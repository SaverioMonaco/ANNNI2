import numpy as np
import pennylane as qml
import os
from typing import Tuple, List

class state:
    def __init__(self, shapes : np.ndarray, tensors : np.ndarray):
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


class mps:
    def __init__(self, folder : str = '../tensor_data/'):
        self.path = folder

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
        self.hs = np.unique(hs_shape)
        self.ks = np.unique(ks_shape)

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

    def get_H(self, h : float, k : float):
        """
        Set up Hamiltonian
        """

        # Interaction of spins with magnetic field
        H = - h * qml.PauliZ(0)
        for i in range(1, self.L):
            H = H - h * qml.PauliZ(i)

        # Interaction between spins (neighbouring):
        for i in range(0, self.L - 1):
            H = H + (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))

        # Interaction between spins (next-neighbouring):
        for i in range(0, self.L - 2):
            H = H + (-1) * k * (qml.PauliX(i) @ qml.PauliX(i + 2))

        return H       
        