import argparse
import numpy as np
import os
import ANNNI.annni as ANNNI
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ANNNI.general as general 

folder = '../runs/'

parser = argparse.ArgumentParser(description='Plot the Phase Map of a trained QCNN on the ANNNI model. Phases: Ferromagnetic, Paramagnetic, Antiphase, Floatingphase')

parser.add_argument('--load', type=str, default='folder', metavar='FOLDER',
                    help='Name of the folder with the QCNN parameters')

parser.add_argument('--save', type=str, default='prediction', metavar='FILE',
                    help='Name of the svg file')

parser.add_argument('--title', type=str, default='Predicted labels', metavar='STR',
                    help='Title of the plot')

args = parser.parse_args()

PREDICTIONS = np.loadtxt(f'{folder}{args.load}/predict')

side = np.sqrt(PREDICTIONS.shape[0]).astype(int)

# cheat the system creating a fake mps class
class fakemps:
    def __init__(self, side, kappa_max = 1, h_max = 2):
        self.ks,  self.hs  = np.linspace(0,kappa_max,side), np.linspace(0,h_max,side)
        self.kappa_max, self.h_max = kappa_max, h_max

mps = fakemps(side)

# Color for representing plots with
#   3 Phases (Ferromagnetic, Paramagnetic, Antiphase)
#   4 Phases : 3 Phases + Floating Phase
col3 = [[0.456, 0.902, 0.635, 1],
                [0.400, 0.694, 0.800, 1],
                [0.922, 0.439, 0.439, 1]]

col4 = [[0.456, 0.902, 0.635, 1],
                [0.400, 0.694, 0.800, 1],
                [1.000, 0.514, 0.439, 1],
                [0.643, 0.012, 0.435, 1]]

col8 = [[0.5, 0.9, 0.6, 1],
                [0.4, 0.7, 0.8, 1],
                [1.0, 0.5, 0.4, 1],
                [0.9, 0.9, 0.0, 1],
                [0.5/2, 0.9/2, 0.6/2, 1],
                [0.4/2, 0.7/2, 0.8/2, 1],
                [1.0/2, 0.5/2, 0.4/2, 1],
                [0.9/2, 0.9/2, 0.0/2, 1]]

# Relative colormaps
cm3 = ListedColormap(col3, name='color3')
cm4 = ListedColormap(col4, name='color4')
cm8 = ListedColormap(col8, name='color8')

def run():
    ARGPREDICTIONS = np.argmax(PREDICTIONS, axis=1)

    cmap = cm3
    # If it detected a 4th class, the colormap is different
    if len(np.unique(ARGPREDICTIONS)) == 4:
        cmap = cm4 

    general.plot_layout(mps, False, True, True, args.title, figure_already_defined = False)
    plt.imshow(np.flip(np.reshape(ARGPREDICTIONS, (side, side)), axis=0), cmap=cmap)
    plt.savefig(f'{folder}{args.load}/{args.save}')


if __name__ == "__main__":
    run()