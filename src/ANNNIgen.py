import numpy as np 
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Union, Tuple

####################
# TRANSITION LINES #
####################

def paraanti(k : Union[float, NDArray]) -> Union[float, NDArray]:
    """Paramagnetic-Antiphase line

    Function for the Paramagnetic-Antiphase transition line.
    Defined for k >= 0.5.
    For k <= 0.5 it returns something close to 0

    Parameters
    ----------
    k : float or NDArray
        Value(s) of k

    Returns
    -------
    float or NDArray
        Value(s) of h at the transition point
    """

    # If input value is a numpy array, we have to set
    # the input k values such that the ones outside its 
    # domain do not lead to runtime errors
    if not (isinstance(k, int) or isinstance(k, float)):
        # To avoid runtime error
        k[k <= .5] = .5 + 1e-4

    return 1.05 * np.sqrt((k - 0.5) * (k - 0.1))


def paraferro(k : Union[float, NDArray]) -> Union[float, NDArray]:
    """Paramagnetic-Ferromagnetic line

    Function for the Paramagnetic-Ferromagnetic transition line.
    Defined for k <= 0.5.
    For k > 0.5 it returns something close to 0

    Parameters
    ----------
    k : float or NDArray
        Value(s) of k

    Returns
    -------
    float or NDArray
        Value(s) of h at the transition point
    """

    # If input value is a numpy array, we have to set
    # the input k values such that the ones outside its 
    # domain do not lead to runtime errors
    if not (isinstance(k, int) or isinstance(k, float)):
        # To avoid runtime error
        k[k >= .5] = .5
        k[k <= 0 ] = 1e-4 # 0 is another problematic point

    return ((1 - k) / k) * (1 - np.sqrt((1 - 3 * k + 4 * k * k) / (1 - k)))


def b1(k : Union[float, NDArray]) -> Union[float, NDArray]:
    """Floating-phase line

    Function for the Floating-phase line.
    Defined for k >= 0.5.
    For k < 0.5 it returns something close to 0

    Parameters
    ----------
    k : float or NDArray
        Value(s) of k

    Returns
    -------
    float or NDArray
        Value(s) of h at the transition point
    """

    # If input value is a numpy array, we have to set
    # the input k values such that the ones outside its 
    # domain do not lead to runtime errors
    if not (isinstance(k, int) or isinstance(k, float)):
        k[k < .5] = .5 # not defined for k < 0.5

    return 1.05 * (k - 0.5)


def peshel_emery(k : Union[float, NDArray]) -> Union[float, NDArray]:
    """Peshel-Emery line

    Function for the Peshel-Emery line.
    Defined for k <= 0.5.

    Parameters
    ----------
    k : float or NDArray
        Value(s) of k

    Returns
    -------
    float or NDArray
        Value(s) of h at the transition point
    """

    # If input value is a numpy array, we have to set
    # the input k values such that the ones outside its 
    # domain do not lead to runtime errors
    if not (isinstance(k, int) or isinstance(k, float)):
        # To avoid Runtime error
        k[k < 1e-4] = 1e-4
        # This function gets high values very quickly
        y = (1 / (4 * k)) - k
        # I am capping them to 2
        # TODO: If we investigate a region for h > 2, this cap will become
        #       visible in the plots
        y[y > 2] = 2

        return y

    y = (1 / (4 * k)) - k

    return y

def get_labels(h : float, k : float) -> Tuple[int,int]:
    """Get labels from h and k

    Get the phase from h and k values. Get the corresponding phase
    when considering or not the floating phase

    Parameters
    ----------
    h : float
        h value (y)
    k : float
        k value (x)

    Returns
    -------
    Tuple[int, int]
        (label (3phases), label (4phases))
    """

    # CASE 1: x=0 axis
    # Added this case because of 0 encountering in division
    if k == 0:
        if h <= 1:
            # Ferromagnetic, Ferromagnetic
            return 0, 0
        else:
            # Paramagnetic, Paramagnetic
            return 1, 1
    # CASE 2: Left side (Paramagnetic - Ferromagnetic)
    elif k < +.5:
        if h <= paraferro(k):
            # Ferromagnetic, Ferromagnetic
            return 0, 0
        else:
            # Paramagnetic, Paramagnetic
            return 1, 1
    # CASE 3: Right side (Antiphase - Floating Phase - Paramagnetic)
    else:
        if h <= paraanti(k):
            if h <= b1(k):
                # Antiphase, Antiphase
                return 2, 2
            else:
                # Antiphase, Floating Phase
                return 2, 3
        else:
            # Paramagnetic, Paramagnetic
            return 1, 1
        
def getlines(mpsclass, func: Callable, xrange: List[float], res: int = 100, **kwargs):
    """
    Plot function func from xrange[0] to xrange[1]
    This function uses the Hamiltonians class to plot the function 
    according to the ranges of its parameters

    Parameters
    ----------
    Hs : hamiltonians.hamiltonian
        Custom Hamiltonian class
    func : function
        Function to plot, usually:
        > general.paraanti : Transition line between paramagnetic phase and antiphase;
        > general.paraferro : Transition line between paramagnetic phase and ferromagnetic phase;
        > general.b1 : Pseudo-transition line inside the antiphase subspace;
        > general.peshel_emery :Peshel Emery Line.
    """

    # Get information from vqeclass for plotting
    # (func needs to be resized)
    side_x = len(mpsclass.ks)
    side_y = len(mpsclass.hs)
    max_x  = max(mpsclass.ks)

    yrange = [0, max(mpsclass.hs)]
    
    xs = np.linspace(xrange[0], xrange[1], res)
    ys = func(xs)

    ys[ys > yrange[1]] = yrange[1]
    
    corrected_xs = (side_x * xs / max_x - 0.5)

    plt.plot(corrected_xs, side_y - ys * side_y / yrange[1] - 0.5, **kwargs)

####################
#  VISUALIZATION   #
####################

def plot_layout(mpsclass, pe_line, phase_lines, floating, title, haxis = True, figure_already_defined = False):
    """
    Many plotting functions here have the same layout, this function will be called inside the others
    to have a standard layout

    Parameters
    ----------
    Hs : hamiltonians.hamiltonian
        Custom hamiltonian class, it is needed to set xlim and ylim and ticks
    pe_line : bool
        if True plots Peshel Emery line
    phase_lines : bool
        if True plots the phase transition lines
    title : str
        Title of the legent of the plot
    figure_already_defined : bool
        if False it calls the plt.figure function
    """

    if not figure_already_defined:
        plt.figure(figsize=(8, 6), dpi=80)

    # Set the axes according to the Hamiltonian class
    if haxis:
        plt.ylabel(r"$h$", fontsize=24)
    plt.xlabel(r"$\kappa$", fontsize=24)
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    n_kappas, n_hs = len(mpsclass.ks), len(mpsclass.hs)
    kappa_max, h_max = max(mpsclass.ks), max(mpsclass.hs)
    ticks_x = [-.5 , n_kappas/4 - .5, n_kappas/2 - .5 , 3*n_kappas/4 - .5, n_kappas - .5]
    ticks_y = [-.5 , n_hs/4 - .5, n_hs/2 - .5 , 3*n_hs/4 - .5, n_hs - .5]
    plt.xticks(
        ticks= ticks_x,
        labels=[np.round(k * kappa_max  / 4, 2) for k in range(0, 5)],
    )
    if haxis:
        plt.yticks(
            ticks=ticks_y,
            labels=[np.round(k * h_max / 4, 2) for k in range(4, -1, -1)],
        )
    else: 
        plt.yticks(ticks=[])

    if pe_line:
        getlines(mpsclass, peshel_emery, [0, 0.5], res=100, color = "blue", alpha=1, ls = '--', dashes=(4,5), label = 'Peshel-Emery line')
        
    if floating:
        getlines(mpsclass, b1, [0.5, kappa_max], res=100, color = "blue", alpha=1, ls = '--', dashes=(4,5), label = 'Floating Phase line')
    
    if phase_lines:
        getlines(mpsclass, paraanti, [0.5, kappa_max], res=100, color = "red", label = 'Phase-transition\n lines')
        getlines(mpsclass, paraferro, [0, 0.5], res=100, color = "red")
    
    if len(title) > 0:
        leg = plt.legend(
                bbox_to_anchor=(1, 1),
                loc="upper right",
                fontsize=16,
                facecolor="white",
                markerscale=1,
                framealpha=0.9,
                title=title,
                title_fontsize=16,
            )
    
    plt.tight_layout()
