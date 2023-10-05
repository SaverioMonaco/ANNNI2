import numpy as np 


####################
# TRANSITION LINES #
####################

def paraanti(x):
    return 1.05 * np.sqrt((x - 0.5) * (x - 0.1))


def paraferro(x):
    return ((1 - x) / x) * (1 - np.sqrt((1 - 3 * x + 4 * x * x) / (1 - x)))


def b1(x):
    return 1.05 * (x - 0.5)


def peshel_emery(x):
    y = (1 / (4 * x)) - x

    y[y > 2] = 2
    return y

def get_labels(h, k):
    if k == 0:
        # Added this case because of 0 encountering in division
        if h <= 1:
            return 0, 0
        else:
            return 1, 1
    elif k > -.5:
        # Left side (yes it is the left side, the phase plot is flipped due to x axis being from 0 to - kappa_max)
        if h <= paraferro(-k):
            return 0, 0
        else:
            return 1, 1
    else:
        # Right side
        if h <= paraanti(-k):
            if h <= b1(-k):
                return 2, 2
            else:
                return 2, 3
        else:
            return 1, 1