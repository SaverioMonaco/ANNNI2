import argparse
import numpy as np
import os
import ANNNIstates as ANNNI
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ANNNIgen 
import tqdm

folder = '../runs/'

parser = argparse.ArgumentParser(description='Predict the Phase Map of a trained QCNN on the ANNNI model. Phases: Ferromagnetic, Paramagnetic, Antiphase, Floatingphase')

parser.add_argument('--load', type=str, default='folder', metavar='FOLDER',
                    help='Name of the folder with the QCNN parameters')

parser.add_argument('--data', type=str, default='20/', metavar='FOLDER',
                    help='Name of the folder with the MPS data')

parser.add_argument('--gpu', type=bool, default=False, metavar='GPU',
                    help='enables training through GPU')

args = parser.parse_args()

PARAMS = np.loadtxt(f'{folder}{args.load}/params')

def run(args):
    TT = ANNNI.mps(folder = f'../tensor_data/{args.data}', gpu=args.gpu)
    TT.qcnn.PARAMS = PARAMS

    # We skip the training
    # TT.train_enc(epochs=args.epochs, train_index=args.point, lr = args.lr)
    
    progress = tqdm.tqdm(range(len(TT.MPS)))
    index = 0

    if os.path.isfile(f'{folder}{args.load}/predict'):
        print('FILE FOUND')
        PREDICTIONS = np.loadtxt(f'{folder}{args.load}/predict')

        for _ in PREDICTIONS:
            index += 1
            progress.update(1)
    else: 
        print('FILE NOT FOUND')

    while index < len(TT.MPS):
        state = TT.MPS[index].towave()
        prediction = TT.qcnn.j_q_circuit(state, TT.qcnn.PARAMS)
        if index == 0:
            PREDICTIONS = prediction
        else: 
            PREDICTIONS = np.vstack((PREDICTIONS, prediction))
        np.savetxt(f'{folder}{args.load}/predict', PREDICTIONS)
        progress.update(1)
        index += 1

if __name__ == "__main__":
    run(args)