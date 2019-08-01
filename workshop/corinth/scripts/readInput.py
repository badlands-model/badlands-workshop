##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling application.    ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module defines several functions used to force Badlands simulation with external
processes related to climate, tectonic and sea level.
"""
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import os
import math
import h5py
import errno
import pandas
import numpy
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
from scipy.interpolate import RectBivariateSpline
import cmocean as cmo
from matplotlib import cm
from pylab import rcParams
from scipy import signal
import scipy.spatial as spatial

import re
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from pylab import rcParams


class readInput:
    """
    This class is used to visualise forcing conditions.
    """
    def __init__(self):
        """
        Initialization function.
        """

        self.X = None
        self.Y = None
        self.Z = None
        self.disp = None
        self.disp3D = None
        self.rain = None
        self.erolays = None
        self.thicklays = None

        self.dx = None
        self.nx = None
        self.ny = None

        self.faultsXY = None
        self.nbfault = None

        return

    def readDEM(self, inDEM):
        """
        Get the initial regular DEM.

        Parameters
        ----------

        variable: inDEM
            Name of the CSV topographic file.
        """

        xyz = pandas.read_csv(str(inDEM), sep=r'\s+', engine='c', header=None, na_filter=False, \
                                   dtype=numpy.float, low_memory=False)
        self.X = xyz.values[:,0]
        self.dx = self.X[1] - self.X[0]
        self.Y = xyz.values[:,1]

        self.nx = int((self.X.max()-self.X.min())/self.dx + 1)
        self.ny = int((self.Y.max()-self.Y.min())/self.dx + 1)

        if self.nx*self.ny != len(self.X):
            raise ValueError('Check your input file the size of the grid is not right.')

        self.Z = numpy.reshape(xyz.values[:,2],(self.nx, self.ny),order='F')

        return

    def readRain(self, inRain):
        """
        Get the initial regular Rain grid.

        Parameters
        ----------

        variable: inRain
            Name of the CSV rain file.
        """

        rain = pandas.read_csv(str(inRain), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        self.rain = numpy.reshape(rain.values,(self.nx, self.ny),order='F')

        return

    def readDisp(self, inTec):
        """
        Get the initial regular vertical tectonic grid.

        Parameters
        ----------

        variable: inTec
            Name of the CSV vertical tectonic file.
        """

        tec = pandas.read_csv(str(inTec), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        self.disp = numpy.reshape(tec.values,(self.nx, self.ny),order='F')

        return

    def readDisp3D(self, inDisp):
        """
        Get the initial regular 3D displacements grid.

        Parameters
        ----------

        variable: inDisp
            Name of the CSV 3D displacements file.
        """

        disp = pandas.read_csv(str(inDisp), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        dX = numpy.reshape(disp.values[:,0],(self.nx, self.ny),order='F')
        dY = numpy.reshape(disp.values[:,1],(self.nx, self.ny),order='F')
        dZ = numpy.reshape(disp.values[:,2],(self.nx, self.ny),order='F')

        self.disp3D = []
        self.disp3D.append(dX)
        self.disp3D.append(dY)
        self.disp3D.append(dZ)

        return

    def readEroLay(self, inEro,id=0):
        """
        Get the initial regular erodibilty layers grid.

        Parameters
        ----------

        variable: inEro
            Name of the CSV erodibilty layers file.

        variable: id
            Erodibilty layer index (starts at 0).
        """

        if self.erolays is None:
            self.erolays = []

        erolay = pandas.read_csv(str(inEro), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        ero = numpy.reshape(erolay.values,(self.nx, self.ny),order='F')
        self.erolays.append(ero)

        return

    def readThickLay(self, inThick,id=0):
        """
        Get the initial regular thickness layers grid.

        Parameters
        ----------

        variable: inThick
            Name of the CSV thickness layers file.

        variable: id
            Erodibilty layer index (starts at 0).
        """

        if self.thicklays is None:
            self.thicklays = []

        thlay = pandas.read_csv(str(inThick), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        th = numpy.reshape(thlay.values,(self.nx, self.ny),order='F')
        self.thicklays.append(th)

        return

    def readFault(self, inFault):
        """
        Get the faults.

        Parameters
        ----------

        variable: inFault
            Name of the CSV fault file.
        """

        with open(inFault) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        nbfault = len(content)

        self.faultsXY = []
        for l in range(nbfault):
            tmp = content[l]
            pts = tmp.count(' ')+1
            tmpXY = numpy.zeros((int(pts/2),2))
            results = [w for w in re.findall('[\d.]+', tmp) if w]
            t = 0
            for k in range(int(pts/2)):
                tmpXY[k,0] = results[t]
                tmpXY[k,1] = results[t+1]
                t += 2
            self.faultsXY.append(tmpXY)

            self.nbfault = nbfault

        return

    def plotInputGrid(self,data=None,title='Title',mind=0,maxd=100,color=None,fault=False,figsave=None):
        """
        Plot the initial forcing grids.

        Parameters
        ----------

        variable: data
            Data to be plotted.

        variable: title
            Plot title.

        variable: mind,maxd
            Extend of colorbar dataset.

        variable: color
            Colorbar.

        variable: figsave
            Saved figure name.
        """

        rcParams['figure.figsize'] = (9,6)
        rcParams['font.size'] = 8

        dataExtent = [numpy.amin(self.X), numpy.amax(self.X), numpy.amin(self.Y), numpy.amax(self.Y)]
        ax=plt.gca()
        im = ax.imshow(numpy.flipud(data.T),interpolation='nearest',cmap=color,vmin=mind, vmax=maxd,
                       extent=dataExtent)
        plt.title(title)

        xx = numpy.reshape(self.X,(self.nx, self.ny),order='F')
        yy = numpy.reshape(self.Y,(self.nx, self.ny),order='F')

        if fault:
            plt.contour(xx, yy, self.Z, (0,), colors='k', linewidths=1,linestyles='dashed')
            for f in range(self.nbfault):
                ax.plot(self.faultsXY[f][:,0],self.faultsXY[f][:,1],'k-',lw=2, zorder=3)
        else:
            plt.contour(xx, yy, self.Z, (0,), colors='k', linewidths=2)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)
        
        plt.colorbar(im,cax=cax)
        plt.show()

        if figsave is not None:
            fig.savefig(figsave)

        plt.close()

        return

    def readSea(self,seafile):
        """
        Plot sea level curve.

        Parameters
        ----------
        variable: seafile
            Absolute path of the sea-lelve data.
        """

        df=pandas.read_csv(seafile, sep=r'\s+',header=None)
        SLtime,sealevel = df[0],df[1]

        rcParams['figure.figsize'] = (6,3)
        rcParams['font.size'] = 8

        ax=plt.gca()
        plt.title('Sea level')

        ax.grid(True)
        ax.set_ylabel('Elevation [m]', fontsize=11)
        ax.set_xlabel('Time [y]', fontsize=10)

        ax.plot(SLtime, sealevel, color='#4286f4',linewidth='2.5')
        plt.show()
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.close()

        return
