##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReefCore synthetic coral reef core model app.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set the rules for carbonate rules.
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from pylab import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def plot_depth_control(xrange=None, fct=None, title=None, xlabel='range',
                       ylabel='function value', color='b',size=(8,4),
                       fname=None):
    '''
    Plotting function for depth control

    Parameters:
    ----------

    variable : xrange
        Extent of the range along the X-axis (numpy array)

    variable : fct
        Value of the rule for the given xrange (numpy array)

    variable : title
        Title of the plot

    variable : xlabel,ylabel
        Label for both axis (string)

    variable : color
        Color of the line

    variable: size
        Figure size

    variable: fname
        Name of the filename to write without extension
    '''

    rcParams['figure.figsize'] = size
    ax=plt.gca()
    ax.plot(xrange, fct, color, linewidth=4)
    plt.grid(b=True, which='both', color='0.35',linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.close()

    # Write membership function
    if fname is not None:
        nameCSV = 'depthcontrol1'
        df = pd.DataFrame({'X':depth,'Y':shallow})
        df.to_csv(str(fname)+'.csv',columns=['X', 'Y'], sep=' ', index=False ,header=0)

    return
