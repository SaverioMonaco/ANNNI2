import argparse
import numpy as np
import os
import ANNNIstates as ANNNI
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ANNNIgen 
import tqdm

folder = '../runs/'

parser = argparse.ArgumentParser(description='Predict the Phase Map of a trained AUTOENCODER on the ANNNI model. Phases: Ferromagnetic, Paramagnetic, Antiphase, Floatingphase')

parser.add_argument('--load', type=str, default='folder', metavar='FOLDER',
                    help='Name of the folder with the AUTOENCODER parameters')

parser.add_argument('--data', type=str, default='20/', metavar='FOLDER',
                    help='Name of the folder with the MPS data')

parser.add_argument('--gpu', type=bool, default=False, metavar='GPU',
                    help='enables training through GPU')

args = parser.parse_args()

PARAMS = np.loadtxt(f'{folder}{args.load}/params')

def run(args):
    TT = ANNNI.mps(folder = f'../tensor_data/{args.data}', gpu=args.gpu)
    TT.enc.PARAMS = PARAMS

    # We skip the training
    # TT.train_enc(epochs=args.epochs, train_index=args.point, lr = args.lr)
    
    progress = tqdm.tqdm(range(len(TT.MPS)))

    if os.path.isfile(f'{folder}{args.load}/predict'):
        print('FILE FOUND')
        COMPRESSIONS = np.loadtxt(f'{folder}{args.load}/predict')
    else: 
        print('FILE NOT FOUND')
        COMPRESSIONS = np.array([])

    index = 0
    for _ in COMPRESSIONS:
        index += 1
        progress.update(1)
    

    while index < len(TT.MPS):
        state = TT.MPS[index].towave()
        COMPRESSIONS = np.append(COMPRESSIONS, TT.enc.j_get_loss(state, TT.enc.PARAMS))
        np.savetxt(f'{folder}{args.load}/predict', COMPRESSIONS)
        progress.update(1)
        index += 1

if __name__ == "__main__":
    run(args)