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
import os
import numpy
import pandas
from scipy.interpolate import RegularGridInterpolator


class resizeInput:
    """
    This class is used to quickly build higher resolution input data file from pre-existing ones.

    Parameters
    ----------
    variable : requestedSpacing
        Required space interval for the new grid (in metres).
    """
    def __init__(self, requestedSpacing = 100):
        """
        Initialization function which takes the requested resolution to use for regridding.

        Parameters
        ----------

        variable: requestedSpacing
            Regridding resolution in metres.
        """

        self.res = requestedSpacing

        self.X = None
        self.Y = None

        self.dx = None
        self.nx = None
        self.ny = None

        self.xgrid = None
        self.ygrid = None

        self.xi = None
        self.yi = None

        return

    def regridDEM(self, inDEM, outDEM):
        """
        Convert the initial regular DEM to the requested resolution.

        Parameters
        ----------

        variable: inDEM
            Name of the CSV topographic file to regrid.

        variable: outDEM
            Name of the new CSV topographic file.
        """

        xyz = pandas.read_csv(str(inDEM), sep=r'\s+', engine='c', header=None, na_filter=False, \
                                   dtype=numpy.float, low_memory=False)
        self.X = xyz.values[:,0]
        self.dx = self.X[1] - self.X[0]
        if self.res >= self.dx:
            print 'Data spacing: ',self.dx
            print 'Requested spacing: ',self.res
            raise ValueError('The requested resolution is lower than the existing one.')
        self.Y = xyz.values[:,1]

        self.nx = int((self.X.max()-self.X.min())/self.dx + 1)
        self.ny = int((self.Y.max()-self.Y.min())/self.dx + 1)
        if self.nx*self.ny != len(self.X):
            raise ValueError('Check your input file the size of the grid is not right.')

        Z = numpy.reshape(xyz.values[:,2],(self.nx, self.ny),order='F')

        self.xgrid = numpy.arange(self.X.min(),self.X.max()+self.dx,self.dx)
        self.ygrid = numpy.arange(self.Y.min(),self.Y.max()+self.dx,self.dx)

        RGI_function = RegularGridInterpolator((self.xgrid, self.ygrid), Z)

        ngridX = numpy.arange(self.X.min(),self.X.max()+self.res,self.res)
        ngridY = numpy.arange(self.Y.min(),self.Y.max()+self.res,self.res)
        self.xi, self.yi = numpy.meshgrid(ngridX, ngridY)

        zi = RGI_function((self.xi.flatten(),self.yi.flatten()))

        df = pandas.DataFrame({'X':self.xi.flatten(),'Y':self.yi.flatten(),'Z':zi.flatten()})
        df.to_csv(str(outDEM),columns=['X', 'Y', 'Z'], sep=' ', index=False ,header=0)

        return

    def regridRain(self, inRain, outRain):
        """
        Convert the initial regular Rain grid to the requested resolution.

        Parameters
        ----------

        variable: inRain
            Name of the CSV rain file to regrid.

        variable: outRain
            Name of the new CSV rain file.
        """

        rain = pandas.read_csv(str(inRain), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        R = numpy.reshape(rain.values,(self.nx, self.ny),order='F')

        RGI_function = RegularGridInterpolator((self.xgrid, self.ygrid), R)

        ri = RGI_function((self.xi.flatten(),self.yi.flatten()))

        df = pandas.DataFrame({'R':ri.flatten()})
        df.to_csv(str(outRain),columns=['R'], sep=' ', index=False ,header=0)

        return

    def regridTecto(self, inTec, outTec):
        """
        Convert the initial regular vertical tectonic grid to the requested resolution.

        Parameters
        ----------

        variable: inTec
            Name of the CSV vertical tectonic file to regrid.

        variable: outTec
            Name of the new CSV vertical tectonic  file.
        """

        tec = pandas.read_csv(str(inTec), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        T = numpy.reshape(tec.values,(self.nx, self.ny),order='F')

        RGI_function = RegularGridInterpolator((self.xgrid, self.ygrid), T)

        ti = RGI_function((self.xi.flatten(),self.yi.flatten()))

        df = pandas.DataFrame({'T':ti.flatten()})
        df.to_csv(str(outTec),columns=['T'], sep=' ', index=False ,header=0)

        return

    def regridDisp(self, inDisp, outDisp):
        """
        Convert the initial regular 3D displacements grid to the requested resolution.

        Parameters
        ----------

        variable: inDisp
            Name of the CSV 3D displacements file to regrid.

        variable: outDisp
            Name of the new CSV 3D displacements file.
        """

        disp = pandas.read_csv(str(inDisp), sep=r'\s+', engine='c',
                               header=None, na_filter=False, dtype=numpy.float, low_memory=False)

        dX = numpy.reshape(disp.values[:,0],(self.nx, self.ny),order='F')
        dY = numpy.reshape(disp.values[:,1],(self.nx, self.ny),order='F')
        dZ = numpy.reshape(disp.values[:,2],(self.nx, self.ny),order='F')

        RGI_function1 = RegularGridInterpolator((self.xgrid, self.ygrid), dX)
        RGI_function2 = RegularGridInterpolator((self.xgrid, self.ygrid), dY)
        RGI_function3 = RegularGridInterpolator((self.xgrid, self.ygrid), dZ)

        dXi = RGI_function1((self.xi.flatten(),self.yi.flatten()))
        dYi = RGI_function2((self.xi.flatten(),self.yi.flatten()))
        dZi = RGI_function3((self.xi.flatten(),self.yi.flatten()))

        df = pandas.DataFrame({'dX':dXi.flatten(),'dY':dYi.flatten(),'dZ':dZi.flatten()})
        df.to_csv(str(outDisp),columns=['dX', 'dY', 'dZ'], sep=' ', index=False ,header=0)

        return
