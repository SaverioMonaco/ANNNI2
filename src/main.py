
import argparse
import datetime
import numpy as np
import os
import ANNNIstates as ANNNI

folder = '../runs/'
now = datetime.datetime.now()
default_name = f'{now.month}-{now.day}_{now.hour}:{now.minute}:{now.second}'

parser = argparse.ArgumentParser(description='Training of QCNN on the ANNNI model. Phases: Ferromagnetic, Paramagnetic, Antiphase, Floatingphase')

parser.add_argument('--save', type=str, default=default_name, metavar='FOLDER',
                    help='Name of the folder to store the results')

parser.add_argument('--gpu', type=bool, default=False, metavar='GPU',
                    help='enables training through GPU')

parser.add_argument('--load', type=str, default='12/', metavar='FOLDER',
                    help='Name of the folder with the MPS data')

parser.add_argument('--analytical', type=bool, default=False, metavar='ANALYTICAL', 
                    help='Only use analytical data points (in the h=0 OR k=0 subspace)')

parser.add_argument('--samples_per_class', type=int, default=10, metavar='N_SAMPLES', 
                    help='Number of input states for each class to load at the same time')

parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS', 
                    help='Number of epochs for each generation')

parser.add_argument('--generations', type=int, default=4, metavar='GEN', 
                    help='Number of generations')

parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='Learning rate of the optimizer')

args = parser.parse_args()

def run(args):
    TT = ANNNI.mps(folder = f'../tensor_data/{args.load}', gpu=args.gpu)
    TT.train_rotate(labels4 = True, analytical = args.analytical,
                     samples_per_class = args.samples_per_class,
                     epochs = args.epochs, generations = args.generations,
                     lr = args.lr,
                     show_samples = False)
    
    if not os.path.exists(f'{folder}{args.save}'):
        os.makedirs(f'{folder}{args.save}')

    TT.qcnn.save_params(f'{folder}{args.save}/params')

    predictions = TT.predict(save = f'{folder}{args.save}/predict.svg')
    np.savetxt(f'{folder}{args.save}/predict', predictions)

if __name__ == "__main__":
    run(args)
