
import argparse
import datetime
import numpy as np
import os
import ANNNI.annni as ANNNI

folder = '../runs/'
now = datetime.datetime.now()
default_name = f'ENC{now.month}-{now.day}_{now.hour}:{now.minute}:{now.second}'

parser = argparse.ArgumentParser(description='Training of QCNN on the ANNNI model. Phases: Ferromagnetic, Paramagnetic, Antiphase, Floatingphase')

parser.add_argument('--save', type=str, default=default_name, metavar='FOLDER',
                    help='Name of the folder to store the results')

parser.add_argument('--gpu', type=bool, default=False, metavar='GPU',
                    help='enables training through GPU')

parser.add_argument('--load', type=str, default='12/', metavar='FOLDER',
                    help='Name of the folder with the MPS data')

parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS', 
                    help='Number of epochs for each generation')

parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='Learning rate of the optimizer')

parser.add_argument('--point', type=int, default=0, metavar='INDEX', 
                    help='Index of the single training point')

parser.add_argument('--serial', type=bool, default=False, metavar='BOOL',
                    help='Compute the compression using the jit function (True) or vmap jit function (False)')


args = parser.parse_args()

def run(args):
    TT = ANNNI.mps(folder = f'../tensor_data/{args.load}', gpu=args.gpu)
    TT.train_enc(epochs=args.epochs, train_index=args.point, lr = args.lr)
    
    if not os.path.exists(f'{folder}{args.save}'):
        os.makedirs(f'{folder}{args.save}')

    TT.enc.save_params(f'{folder}{args.save}/params')

    compressions = TT.compress(save = f'{folder}{args.save}/predict.svg', serial = args.serial, bar = True)
    np.savetxt(f'{folder}{args.save}/predict', compressions)

if __name__ == "__main__":
    run(args)
