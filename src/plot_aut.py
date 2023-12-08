import argparse
import numpy as np
import os
import ANNNIstates as ANNNI
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ANNNIgen 

folder = '../runs/'

parser = argparse.ArgumentParser(description='Plot the Phase Map of a trained AUTOENCODER on the ANNNI model. Phases: Ferromagnetic, Paramagnetic, Antiphase, Floatingphase')

parser.add_argument('--load', type=str, default='folder', metavar='FOLDER',
                    help='Name of the folder with the AUTOENCODER parameters')

parser.add_argument('--save', type=str, default='prediction', metavar='FILE',
                    help='Name of the svg file')

parser.add_argument('--title', type=str, default='Predicted labels', metavar='STR',
                    help='Title of the plot')

args = parser.parse_args()

COMPRESSIONS = np.loadtxt(f'{folder}{args.load}/predict')

side = np.sqrt(COMPRESSIONS.shape[0]).astype(int)

# cheat the system creating a fake mps class
class fakemps:
    def __init__(self, side, kappa_max = 1, h_max = 2):
        self.ks,  self.hs  = np.linspace(0,kappa_max,side), np.linspace(0,h_max,side)
        self.kappa_max, self.h_max = kappa_max, h_max

mps = fakemps(side)


def run():
    ANNNIgen.plot_layout(mps, False, True, True, args.title, figure_already_defined = False)
    plt.imshow(np.flip(np.reshape(COMPRESSIONS, (side, side)), axis=0), cmap='viridis')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    plt.savefig(f'{folder}{args.load}/{args.save}')

if __name__ == "__main__":
    run()