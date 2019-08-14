##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling application.    ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module defines several functions used to compute catchment erosion.
"""
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import os
import math
import h5py
import errno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
from scipy.interpolate import RectBivariateSpline
from matplotlib.path import Path

from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np
import pylab as pl
from descartes import PolygonPatch
from matplotlib.collections import LineCollection

import cmocean as cmo
from pylab import rcParams
import warnings
from scipy import signal
warnings.filterwarnings('ignore')

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from matplotlib import cm

# Import badlands grid generation toolbox
import badlands_companion.hydroGrid as hydr


def two_scales(ax1, time, data0, data1, c1, c2, font):

    ax2 = ax1.twinx()

    ax1.plot(time, data0, color=c1, linewidth=3)
    ax1.set_xlabel('Time [y]',size=font+2)
    ax1.set_ylabel('deposition rate [mm/y]',size=font+2)
    ax1.yaxis.label.set_color(c1)

    ax2.plot(time, data1, color=c2, linewidth=3)
    ax2.set_ylabel('erosion rate [mm/y]',size=font+2)
    ax2.yaxis.label.set_color(c2)

    return ax1, ax2

def temporalPlot(time=None,ero=None,depo=None,colors=None, size=(10,5)):
    """
    Temporal plot.

    Parameters
    ----------

    variable : colors
        Matplotlib color map to use

    variable : size
        Figure size
    """

    if colors is not None:
        c1 = colors[0]
        c2 = colors[-1]
    else:
        c2 = '#1f77b4'
        c1 = '#229649'

    # Define figure size
    fig, ax = plt.subplots(1,figsize=size, dpi=80)
    ax.set_facecolor('#f2f2f3')

    ax1, ax2 = two_scales(ax,time,depo,ero,c1,c2,10)

    ttl = ax.title
    ttl.set_position([.5, 1.05])
    plt.title('Temporal variation of average erosion & deposition rate around reefs')

    color_y_axis(ax1, c1)
    color_y_axis(ax2, c2)
    ax2.plot(time, ero, linewidth=2, c='#4badf2', linestyle='--', label='erosion rate', zorder=0)

    plt.xlim(time.min(),time.max())

    plt.grid()
    plt.show()

    return

def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)

        return None
def temporalGrowth(folder='output',step=None,vtime=None,smooth=None,vrange=None):
    """
    Temporal coral growth analyse function.

    Parameters
    ----------
    variable : folder
        Folder path to Badlands outputs.

    variable : step
        Number of steps in the simulation.

    variable : smooth
        Number of point for interpolation.

    variable : vtime
        Time array.

    variable: vrange
        Extent of the plot along X and Y direction

    """
    eroh = []
    depoh = []

    timestep=vtime[1]-vtime[0]
    for stp in range(1,step+1):
        data = analyseWaveReef(folder='output',timestep=stp)
        data.regridTINdataSet()
        r,c = np.where(data.depreef>0)
        erodepreef = np.zeros(data.dz.shape)
        erodepreef[r,c] = data.dz[r,c]
        dh = data.getDepositedVolume(data=erodepreef,time=1.,erange=vrange,out=True)
        eh = data.getErodedVolume(data=erodepreef,time=1.,erange=vrange,out=True)
        eroh.append(eh)
        depoh.append(dh)

    erorate = np.diff(np.nan_to_num(np.asarray(eroh)))*1000./timestep
    deporate = np.diff(np.nan_to_num(np.asarray(depoh)))*1000./timestep

    erorate[erorate<0.] = 0.
    deporate[deporate<0.] = 0.

    if smooth is None:
        return vtime,erorate,deporate
    elif smooth < len(vtime):
        return vtime,erorate,deporate
    else:
        timenew = np.linspace(vtime.min(),vtime.max(),smooth)
        #erorate_smooth = spline(vtime,erorate,timenew)
        #deporate_smooth = spline(vtime,deporate,timenew)
        fe = interp1d(vtime,erorate, kind='cubic') 
        fd = interp1d(vtime,deporate, kind='cubic')
        erorate_smooth = fe(timenew)
        deporate_smooth = fd(timenew)
        
        erorate_smooth[erorate_smooth<0]=0
        deporate_smooth[deporate_smooth<0]=0
        return timenew,erorate_smooth,deporate_smooth

class analyseWaveReef:
    """
    This class is used to compute wave reef.
    """

    def __init__(self,folder=None, timestep=0, pointXY=None):
        """
        Initialization function.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.

        variable : timestep
            Time step to load.
        """

        self.folder = folder+'/h5'
        self.timestep = timestep
        self.shaded = None
        self.xi = None
        self.yi = None
        self.z = None
        self.wavesed = None
        self.wavestress = None
        self.cumhill = None
        self.depreef = None
        self.dz = None
        self.dataExtent =  None
        self.dx = None

        self.slp = None
        self.aspect = None
        self.hcurv = None
        self.vcurv = None

        self.pointXY = pointXY
        self.hydro = None
        if self.pointXY is not None:
            hydro = hydr.hydroGrid(folder=self.folder, ptXY = pointXY)
            hydro.getCatchment(timestep)
            self.hydro = hydro

        self.regionx = None
        self.regiony = None
        self.regiondz = None
        self.regionz = None

        # Get points on the grid which are within the catchment
        self.concave_hull = None
        self.edge_points = None
        self.xpts = None
        self.ypts = None
        self.rdz = None

        return

    def getDepositedVolume(self, data=None, time=None, erange=None,out=False):
        """
        Get the volume deposited in a specific region

        Parameters
        ----------
        variable: time
            Model duration for the given time step

        variable: erange
            Extent of the box where computation is performed
        """

        posz = np.copy(data)
        if erange is None:
            posz[posz<0.] = 0.
            volsed = np.sum(posz)*(self.dx*self.dx)
            r,c = np.where(posz > 0.001)
            ratesed = -(np.sum(posz/time)*1000./len(r))
        else:
            r1,c1 = np.where(np.logical_and(self.xi>=erange[0],self.xi<=erange[1]))
            r2,c2 = np.where(np.logical_and(self.yi>=erange[2],self.yi<=erange[3]))
            lposz = posz[r2.min():r2.max(),c1.min():c1.max()]
            lposz[lposz<0.] = 0.
            r,c = np.where(lposz > 0.001)
            volsed = np.sum(lposz)*(self.dx*self.dx)
            ratesed = (np.sum(lposz/time)*1000./len(r))


        if out:
            return (np.sum(lposz/time)/len(r))
        else:
            print('Volume of deposited sediment in cubic metres: ',volsed)
            print('Averaged annual deposited sediment volume in cubic metres per year: ',volsed/time)
            print('Average deposition rate in mm per year: ',ratesed)
            return

    def plotdataSet(self, title='Title', data=None, color=None,  crange=None,
                                erange=None, depctr=None, pt=None, ctr='k',size=(8,8)):
        """
        Plot a given dataset from Badlands output

        Parameters
        ----------
        variable: title
            Title of the plot

        variable: data
            Data to plot

        variable: color
            Colormap

        variable: crange
            Range of values for the dataset

        variable: erange
            Extent of the plot along X and Y direction

        variable: depctr
            Deposit contour

        variable: pt
            Add a given point position on the map

        variable: size
            Figure size

        """

        rcParams['figure.figsize'] = size
        ax=plt.gca()

        if erange is not None:
            r1,c1 = np.where(np.logical_and(self.xi>=erange[0],self.xi<=erange[1]))
            r2,c2 = np.where(np.logical_and(self.yi>=erange[2],self.yi<=erange[3]))

            rdata = data[r2.min():r2.max(),c1.min():c1.max()]

            im = ax.imshow(np.flipud(rdata),interpolation='nearest',cmap=color,
                       vmin=crange[0], vmax=crange[1], extent=erange)

            if depctr is not None:
                plt.contour(self.xi[r2.min():r2.max(),c1.min():c1.max()], self.yi[r2.min():r2.max(),c1.min():c1.max()],
                                    data[r2.min():r2.max(),c1.min():c1.max()], depctr, colors=ctr, linewidths=2,zorder=2)

                plt.contour(self.xi[r2.min():r2.max(),c1.min():c1.max()], self.yi[r2.min():r2.max(),c1.min():c1.max()],
                                    self.z[r2.min():r2.max(),c1.min():c1.max()], (0,), colors='w', linewidths=2,zorder=1)
            else:
                plt.contour(self.xi[r2.min():r2.max(),c1.min():c1.max()], self.yi[r2.min():r2.max(),c1.min():c1.max()],
                                    self.z[r2.min():r2.max(),c1.min():c1.max()], (0,), colors=ctr, linewidths=2)

        else:
            im = ax.imshow(np.flipud(data),interpolation='nearest',cmap=color,
                           vmin=crange[0], vmax=crange[1], extent=self.dataExtent)

            if depctr is not None:
                plt.contour(self.xi, self.yi,
                                    data, depctr, colors=ctr, linewidths=2)
            else:
                plt.contour(self.xi, self.yi, self.z, (0,), colors=ctr, linewidths=2)

        if pt is not None:
            plt.plot(pt[0],pt[1], 'o', markersize=10, markerfacecolor='#f442a1',
                     markeredgewidth=1.5, markeredgecolor='k')
        plt.title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.colorbar(im,cax=cax)
        plt.show()
        plt.close()

        return

    def regridTINdataSet(self, smth=2,dx=None):
        """
        Read the HDF5 file for a given time step and build slope and aspect

        Parameters
        ----------
        variable: smth
            Gaussian filter

        variable: dx
            Discretisation value in metres.

        """

        azimuth=315.0
        altitude=45.0

        if not os.path.isdir(self.folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        df = h5py.File('%s/tin.time%s.hdf5'%(self.folder, self.timestep), 'r')
        coords = np.array((df['/coords']))
        cumdiff = np.array((df['/cumdiff']))
        wavesed = np.array((df['/cumwave']))
        x, y, z = np.hsplit(coords, 3)
        wavestress = np.array((df['/waveS']))
        cumhill = np.array((df['/cumhill']))
        reef1 = np.array((df['/depSpecies1']))
        reef2 = np.array((df['/depSpecies2']))

        self.dx = dx
        if dx is None:
            self.dx = (x[1]-x[0])[0]
            #print 'Set dx to:',dx
        dx = self.dx
        nx = int((x.max() - x.min())/dx+1)
        ny = int((y.max() - y.min())/dx+1)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        xi, yi = np.meshgrid(xi, yi)
        xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
        XY = np.column_stack((x,y))
        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        z_vals = z[indices][:,:,0]
        zi = np.average(z_vals,weights=(1./distances), axis=1)

        dz_vals = cumdiff[indices][:,:,0]
        dzi = np.average(dz_vals,weights=(1./distances), axis=1)

        ws_vals = wavesed[indices][:,:,0]
        wsi = np.average(ws_vals,weights=(1./distances), axis=1)

        cumhill_vals = cumhill[indices][:,:,0]
        hilli = np.average(cumhill_vals,weights=(1./distances), axis=1)

        ss_vals = wavestress[indices][:,:,0]
        ssi = np.average(ss_vals,weights=(1./distances), axis=1)

        r1_vals = reef1[indices][:,:,0]
        r1i = np.average(r1_vals,weights=(1./distances), axis=1)

        r2_vals = reef2[indices][:,:,0]
        r2i = np.average(r2_vals,weights=(1./distances), axis=1)

        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0],0]
            dzi[onIDs] = cumdiff[indices[onIDs,0],0]
            wsi[onIDs] = wavesed[indices[onIDs,0],0]
            hilli[onIDs] = cumhill[indices[onIDs,0],0]
            ssi[onIDs] = wavestress[indices[onIDs,0],0]
            r1i[onIDs] = reef1[indices[onIDs,0],0]
            r2i[onIDs] = reef2[indices[onIDs,0],0]

        z = np.reshape(zi,(ny,nx))
        dz = np.reshape(dzi,(ny,nx))
        ws = np.reshape(wsi,(ny,nx))

        hs = np.reshape(hilli,(ny,nx))
        ss = np.reshape(ssi,(ny,nx))
        r1 = np.reshape(r1i,(ny,nx))
        r2 = np.reshape(r2i,(ny,nx))

        reef = r1+r2
        reef[reef>0] = 1

        # Calculate gradient
        Sx, Sy = np.gradient(z)

        rad2deg = 180.0 / np.pi
        slope = 90. - np.arctan(np.sqrt(Sx**2 + Sy**2))*rad2deg
        slp = np.sqrt(Sx**2 + Sy**2)

        aspect = np.arctan2(-Sx, Sy)
        deg2rad = np.pi / 180.0
        shaded = np.sin(altitude*deg2rad) * np.sin(slope*deg2rad) \
                 + np.cos(altitude*deg2rad) * np.cos(slope*deg2rad) \
                 * np.cos((azimuth - 90.0)*deg2rad - aspect)

        shaded = shaded * 255

        self.shaded = shaded
        self.xi = xi
        self.yi = yi
        self.z = z
        self.dz = dz
        self.wavesed = ws
        self.wavestress = ss
        self.cumhill = hs
        self.depreef = reef
        self.dataExtent = [np.amin(xi), np.amax(xi), np.amin(yi), np.amax(yi)]

        # Applying a Gaussian filter
        self.cmptParams(xi,yi,z)
        z_gauss = self.smoothData(z, smth)
        dz_gauss = self.smoothData(dz, smth)
        self.cmptParams(xi, yi, z_gauss)

        return

    def getErodedVolume(self, data=None, time=None, erange=None,out=False):
        """
        Get the volume eroded in a specific region

        Parameters
        ----------
        variable: time
            Model duration for the given time step

        variable: data
            Dataset to plot

        variable: erange
            Extent of the box where computation is performed
        """

        posz = np.copy(data)
        if erange is None:
            posz[posz>0.] = 0.
            volsed = np.sum(posz)*(self.dx*self.dx)
            r,c = np.where(posz < -0.001)
            ratesed = -(np.sum(posz/time)*1000./len(r))
        else:
            r1,c1 = np.where(np.logical_and(self.xi>=erange[0],self.xi<=erange[1]))
            r2,c2 = np.where(np.logical_and(self.yi>=erange[2],self.yi<=erange[3]))
            lposz = posz[r2.min():r2.max(),c1.min():c1.max()]
            lposz[lposz>0.] = 0.
            r,c = np.where(lposz < -0.001)
            volsed = np.sum(lposz)*(self.dx*self.dx)
            ratesed = -(np.sum(lposz/time)*1000./len(r))

        if out:
            return -(np.sum(lposz)/len(r))
        else:
            print('Volume of eroded sediment in cubic metres: ',volsed)
            print('Averaged annual eroded sediment volume in cubic metres per year: ',volsed/time)
            print('Average erosion rate in mm per year: ',ratesed)
            return

    def plotEroElev(self,time=None):
        """
        Plotting erosion rate versus elevation for the catchment.
        """

        r,c = np.where(self.rdz < -0.1)
        rz = np.copy(self.regionz)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        hb = ax.hexbin(rz[r,c].flatten(),-self.rdz[r,c].flatten()/(time/1000.), mincnt = 1, bins='log',
                gridsize = 50, cmap=cmo.cm.thermal, edgecolors = 'w', linewidths = 0.005 , clip_on = True, zorder=3)

        plt.colorbar(hb,orientation='vertical',label='count',fraction=0.03, pad=0.04)
        plt.title('Erosion vs elevation')
        ax.grid(True)
        plt.xlabel('elevation [m]')
        plt.ylabel('erosion rate [mm/y]')
        plt.show()
        plt.close()

        return

    def plotCatchment(self):
        """
        Plotting the catchment.
        """

        margin = .3
        x_min, y_min, x_max, y_max =self.concave_hull.bounds

        rcParams['figure.figsize'] = (8,8)
        ax=plt.gca()
        dataExtent = [self.hydro.rcvX.min()-100., self.hydro.rcvX.max()+100.,
                      self.hydro.rcvY.min()-100., self.hydro.rcvY.max()+100.]

        im = ax.imshow(np.flipud(self.regiondz),interpolation='nearest',cmap=cmo.cm.balance,
                       vmin=-200, vmax=200, extent=dataExtent)

        ax.scatter(self.hydro.rcvX,self.hydro.rcvY,s=1,c='k')

        ax.scatter(self.xpts,self.ypts,s=4,c='r', zorder=2)

        plt.title('Erosion/Deposition in region of interest [m]')

        x_min, y_min, x_max, y_max = self.concave_hull.bounds
        ax.set_xlim([x_min-margin, x_max+margin])
        ax.set_ylim([y_min-margin, y_max+margin])
        patch = PolygonPatch(self.concave_hull, fc='#999999', ec='#000000', fill=False, zorder=1)
        ax.add_patch(patch)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.colorbar(im,cax=cax)
        plt.show()
        plt.close()

        ax=plt.gca()
        # dataExtent = [self.hydro.rcvX.min()-100., self.hydro.rcvX.max()+100.,
        #               self.hydro.rcvY.min()-100., self.hydro.rcvY.max()+100.]

        im = ax.imshow(np.flipud(self.rdz),interpolation='nearest',cmap=cmo.cm.balance,
                       vmin=-200, vmax=200, extent=dataExtent)

        ax.scatter(self.hydro.rcvX,self.hydro.rcvY,s=1,c='k')

        ax.scatter(self.xpts,self.ypts,s=4,c='r', zorder=2)

        plt.title('Erosion in region of interest [m]')

        x_min, y_min, x_max, y_max = self.concave_hull.bounds
        ax.set_xlim([x_min-margin, x_max+margin])
        ax.set_ylim([y_min-margin, y_max+margin])
        patch = PolygonPatch(self.concave_hull, fc='#999999', ec='#000000', fill=False, zorder=1)
        ax.add_patch(patch)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.colorbar(im,cax=cax)
        plt.show()
        plt.close()

        return

    def extractRegion(self,alpha=0.002):
        '''
        Extract catchment points in the TIN

        Parameters
        ----------
        variable: alpha
            Concave hull fitting within the catchment delineation.
        '''

        # We work on the created regular grid interpolated from the TIN (delaunay grid)
        r1,c1 = np.where(np.logical_and(self.xi>self.hydro.rcvX.min()-100.,self.xi<self.hydro.rcvX.max()+100.))
        r2,c2 = np.where(np.logical_and(self.yi>self.hydro.rcvY.min()-100.,self.yi<self.hydro.rcvY.max()+100.))

        self.regionx = self.xi[r2.min():r2.max(),c1.min():c1.max()]
        self.regiony = self.yi[r2.min():r2.max(),c1.min():c1.max()]
        self.regiondz = self.dz[r2.min():r2.max(),c1.min():c1.max()]
        #regionvcurv = self.vcurv[r2.min():r2.max(),c1.min():c1.max()]
        self.regionz = self.z[r2.min():r2.max(),c1.min():c1.max()]

        # Get points on the grid which are within the catchment
        points=[]
        for t in range(len(self.hydro.rcvX)):
            points.append(geometry.shape(geometry.Point(self.hydro.rcvX[t],self.hydro.rcvY[t])))
        self.concave_hull, self.edge_points = self.alpha_shape(points, alpha)
        self.xpts = self.concave_hull.exterior.xy[0]
        self.ypts = self.concave_hull.exterior.xy[1]

        #  Nullify points outside the catchment of interest
        rx,ry = self.regionx.flatten(), self.regiony.flatten()
        mpoints = np.vstack((rx,ry)).T
        poly_verts = np.column_stack((self.xpts,self.ypts))
        path = Path(poly_verts)
        mgrid = path.contains_points(mpoints)
        mgrid = mgrid.reshape(self.regiondz.shape)

        # Apply mask array and nullify depositional area
        r,c = np.where(mgrid == False)
        self.rdz = np.copy(self.regiondz)
        self.rdz[r,c] = 0.
        self.rdz[self.rdz>0] = 0.

        return

    def gaussianFilter(self,sizex,sizey=None,scale=0.333):
        '''
        Generate and return a 2D Gaussian function
        of dimensions (sizex,sizey)

        If sizey is not set, it defaults to sizex
        A scale can be defined to widen the function (default = 0.333)
        '''
        sizey = sizey or sizex
        x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
        g = np.exp(-scale*(x**2/float(sizex)+y**2/float(sizey)))

        return g/g.sum()

    def smoothData(self,dem, smth=2):
        '''
        Calculate the slope and gradient of a DEM
        '''

        gaussZ = np.zeros((dem.shape[0]+6,dem.shape[1]+6))
        gaussZ[3:-3,3:-3] = dem

        f0 = self.gaussianFilter(smth)
        smoothDEM = signal.convolve(gaussZ,f0,mode='valid')

        return smoothDEM[1:-1,1:-1]


    def assignBCs(self,z,nx,ny):
        """
        Pads the boundaries of a grid. Boundary condition pads the boundaries
        with equivalent values to the data margins, e.g. x[-1,1] = x[1,1].
        It creates a grid 2 rows and 2 columns larger than the input.
        """
        Zbc = np.zeros((nx + 2, ny + 2))
        Zbc[1:-1,1:-1] = z

        # Assign boundary conditions - sides
        Zbc[0, 1:-1] = z[0, :]
        Zbc[-1, 1:-1] = z[-1, :]
        Zbc[1:-1, 0] = z[:, 0]
        Zbc[1:-1, -1] = z[:,-1]

        # Assign boundary conditions - corners
        Zbc[0, 0] = z[0, 0]
        Zbc[0, -1] = z[0, -1]
        Zbc[-1, 0] = z[-1, 0]
        Zbc[-1, -1] = z[-1, 0]

        return Zbc

    def cmptParams(self,x,y,Z):
        """
        Define aspect, gradient and horizontal/vertical curvature using a
        quadratic polynomial method.
        """

        # Assign boundary conditions
        Zbc = self.assignBCs(Z,x.shape[0],x.shape[1])

        # Neighborhood definition
        # z1     z2     z3
        # z4     z5     z6
        # z7     z8     z9

        z1 = Zbc[2:, :-2]
        z2 = Zbc[2:,1:-1]
        z3 = Zbc[2:,2:]
        z4 = Zbc[1:-1, :-2]
        z5 = Zbc[1:-1,1:-1]
        z6 = Zbc[1:-1, 2:]
        z7 = Zbc[:-2, :-2]
        z8 = Zbc[:-2, 1:-1]
        z9 = Zbc[:-2, 2:]

        # Compute coefficient values
        dx = x[0,1]-x[0,0]
        zz = z2+z5
        r = ((z1+z3+z4+z6+z7+z9)-2.*(z2+z5+z8))/(3. * dx**2)
        t = ((z1+z2+z3+z7+z8+z9)-2.*(z4+z5+z6))/(3. * dx**2)
        s = (z3+z7-z1-z9)/(4. * dx**2)
        p = (z3+z6+z9-z1-z4-z7)/(6.*dx)
        q = (z1+z2+z3-z7-z8-z9)/(6.*dx)
        u = (5.*z1+2.*(z2+z4+z6+z8)-z1-z3-z7-z9)/9.
        #
        with np.errstate(invalid='ignore',divide='ignore'):
            grad = np.arctan(np.sqrt(p**2+q**2))
            aspect = np.arctan(q/p)
            hcurv = -(r*q**2-2.*p*q*s+t*p**2) / \
                    ((p**2+q**2)*np.sqrt(1+p**2+q**2))
            vcurv = -(r*p**2+2.*p*q*s+t*q**2) /  \
                    ((p**2+q**2)*np.sqrt(1+p**2+q**2))

            self.slp = grad
            self.aspect = aspect
            self.hcurv = hcurv
            self.vcurv = vcurv

            return

    def alpha_shape(self,points, alpha):
        """
        Compute the alpha shape (concave hull) of a set of points.

        @param points: Iterable container of points.
        @param alpha: alpha value to influence the gooeyness of the border. Smaller
                      numbers don't fall inward as much as larger numbers. Too large,
                      and you lose everything!
        """
        if len(points) < 4:
            # When you have a triangle, there is no sense in computing an alpha
            # shape.
            return geometry.MultiPoint(list(points)).convex_hull

        def add_edge(edges, edge_points, coords, i, j):
            """Add a line between the i-th and j-th points, if not in the list already"""
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            edges.add( (i, j) )
            edge_points.append(coords[ [i, j] ])

        coords = np.array([point.coords[0] for point in points])

        tri = Delaunay(coords)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]

            # Lengths of sides of triangle
            a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
            c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

            # Semiperimeter of triangle
            s = (a + b + c)/2.0

            # Area of triangle by Heron's formula
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)

            # Here's the radius filter.
            #print circum_r
            if circum_r < 1.0/alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)

        m = geometry.MultiLineString(edge_points)
        triangles = list(polygonize(m))

        return cascaded_union(triangles), edge_points
