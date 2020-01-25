##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to build simple cross-section from Badlands outputs.
"""

import os
import math
import h5py
import errno
import numpy as np
from pyevtk.hl import gridToVTK

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class stratalMesh:
    """
    Class for creating stratigraphic mesh from Badlands outputs.
    """

    def __init__(self, folder=None):
        """
        Initialization function which takes the folder path to Badlands outputs.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.x = None
        self.y = None
        self.xi = None
        self.yi = None
        self.dx = None
        self.dist = None
        self.dx = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dep = None
        self.th = None
        self.elev = None
        self.timestep = 0

        return

    def loadStratigraphy(self, timestep=0):
        """
        Read the HDF5 file for a given time step.

        Parameters
        ----------
        variable : timestep
            Time step to load.
        """

        self.timestep = timestep

        df = h5py.File('%s/sed.time%s.hdf5'%(self.folder, timestep), 'r')
        coords = np.array((df['/coords']))
        layDepth = np.array((df['/layDepth']))
        layElev = np.array((df['/layElev']))
        layThick = np.array((df['/layThick']))
        x, y = np.hsplit(coords, 2)
        dep = layDepth
        elev = layElev
        th = layThick

        self.dx = x[1]-x[0]
        self.x = x
        self.y = y
        self.nx = int((x.max() - x.min())/self.dx+1)
        self.ny = int((y.max() - y.min())/self.dx+1)
        self.nz = dep.shape[1]
        self.xi = np.linspace(x.min(), x.max(), self.nx)
        self.yi = np.linspace(y.min(), y.max(), self.ny)
        self.dep = dep.reshape((self.ny,self.nx,self.nz))
        self.elev = elev.reshape((self.ny,self.nx,self.nz))
        self.th = th.reshape((self.ny,self.nx,self.nz))

        return

    def buildMesh(self, outfolder='.'):
        """
        Create a vtk unstructured grid based on current time step stratal parameters.

        Parameters
        ----------
        variable : outfolder
            Folder path to store the stratal vtk mesh.
        """

        vtkfile = '%s/stratalMesh.time%s'%(outfolder, self.timestep)

        x = np.zeros((self.nx, self.ny, self.nz))
        y = np.zeros((self.nx, self.ny, self.nz))
        z = np.zeros((self.nx, self.ny, self.nz))
        e = np.zeros((self.nx, self.ny, self.nz))
        h = np.zeros((self.nx, self.ny, self.nz))
        t = np.zeros((self.nx, self.ny, self.nz))

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    x[i,j,k] = self.xi[i]
                    y[i,j,k] = self.yi[j]
                    z[i,j,k] = self.dep[j,i,k]
                    e[i,j,k] = self.elev[j,i,k]
                    h[i,j,k] = self.th[j,i,k]
                    t[i,j,k] = k

        gridToVTK(vtkfile, x, y, z, pointData = {"relative elevation" : e, "thickness" :h, "layer ID" :t})

        return
