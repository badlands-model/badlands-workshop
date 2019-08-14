##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to analyse stratigraphic sequences from Badlands outputs.
"""

import os
import math
import h5py
import errno
import numpy as np
import pandas as pd
from cmocean import cm
import colorlover as cl
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
import scipy.ndimage.filters as filters
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

from pylab import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import cKDTree

import plotly
from plotly import tools
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def interp(scl, r):
    ''' Interpolate a color scale "scl" to a new one with length "r"
        Fun usage in IPython notebook:
        HTML( to_html( to_hsl( interp( cl.scales['11']['qual']['Paired'], 5000 ) ) ) ) '''
    c = []
    SCL_FI = len(scl)-1 # final index of color scale
    # garyfeng:
    # the following line is buggy.
    # r = [x * 0.1 for x in range(r)] if isinstance( r, int ) else r
    r = [x*1.0*SCL_FI/r for x in range(r)] if isinstance( r, int ) else r
    # end garyfeng

    scl = cl.to_numeric( scl )

    def interp3(fraction, start, end):
        ''' Interpolate between values of 2, 3-member tuples '''
        def intp(f, s, e):
            return s + (e - s)*f
        return tuple([intp(fraction, start[i], end[i]) for i in range(3)])

    def rgb_to_hsl(rgb):
        ''' Adapted from M Bostock's RGB to HSL converter in d3.js
            https://github.com/mbostock/d3/blob/master/src/color/rgb.js '''
        r,g,b = float(rgb[0])/255.0,\
                float(rgb[1])/255.0,\
                float(rgb[2])/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        h = s = l = (mx + mn) / 2
        if mx == mn: # achromatic
            h = 0
            s = 0 if l > 0 and l < 1 else h
        else:
            d = mx - mn;
            s =  d / (mx + mn) if l < 0.5 else d / (2 - mx - mn)
            if mx == r:
                h = (g - b) / d + ( 6 if g < b else 0 )
            elif mx == g:
                h = (b - r) / d + 2
            else:
                h = r - g / d + 4

        return (int(round(h*60,4)), int(round(s*100,4)), int(round(l*100,4)))

    for i in r:
        # garyfeng: c_i could be rounded up so scl[c_i+1] will go off range
        #c_i = int(i*math.floor(SCL_FI)/round(r[-1])) # start color index
        #c_i = int(math.floor(i*math.floor(SCL_FI)/round(r[-1]))) # start color index
        #c_i = if c_i < len(scl)-1 else hsl_o

        c_i = int(math.floor(i))
        section_min = math.floor(i)
        section_max = math.ceil(i)
        fraction = (i-section_min) #/(section_max-section_min)

        hsl_o = rgb_to_hsl( scl[c_i] ) # convert rgb to hls
        hsl_f = rgb_to_hsl( scl[c_i+1] )
        #section_min = c_i*r[-1]/SCL_FI
        #section_max = (c_i+1)*(r[-1]/SCL_FI)
        #fraction = (i-section_min)/(section_max-section_min)
        hsl = interp3( fraction, hsl_o, hsl_f )
        c.append( 'hsl'+str(hsl) )

    return cl.to_hsl( c )

def viewSection(width = 800, height = 400, cs = None, dnlay = None,
                rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
    """
    Plot multiple cross-sections data on a graph.
    Parameters
    ----------
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: cs
        Cross-sections dataset.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    nlay = len(cs.secDep)
    colors = cl.scales['9']['div']['BrBG']
    hist = interp( colors, nlay )
    colorrgb = cl.to_rgb( hist )

    trace = {}
    data = []

    trace[0] = Scatter(
        x=cs.dist,
        y=cs.secElev[0],
        mode='lines',
        line=dict(
            shape='spline',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[0])

    for i in range(1,nlay-1,dnlay):
        trace[i] = Scatter(
            x=cs.dist,
            y=cs.secElev[i],
            mode='lines',
            line=dict(
                shape='spline',
                width = linesize,
                color = 'rgb(0,0,0)'
            ),
            opacity=0.5,
            fill='tonexty',
            fillcolor=colorrgb[i]
        )
        data.append(trace[i])

    trace[nlay-1] = Scatter(
        x=cs.dist,
        y=cs.secElev[nlay-1],
        mode='lines',
        line=dict(
            shape='spline',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        ),
        fill='tonexty',
        fillcolor=colorrgb[nlay-1]
    )
    data.append(trace[nlay-1])

    trace[nlay] = Scatter(
        x=cs.dist,
        y=cs.secElev[0],
        mode='lines',
        line=dict(
            shape='spline',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[nlay])

    if rangeX is not None and rangeY is not None:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height,
                showlegend = False,
                xaxis=dict(title='distance [m]',
                            range=rangeX,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks'),
                yaxis=dict(title='elevation [m]',
                            range=rangeY,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks')
        )
    else:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height
        )
    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

def viewSectionProp(width = 8, height = 5, cs = None, dnlay = None, color = None,
                      rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
    """
    Plot stratal stacking pattern colored by proportion depth.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: colors
        Colors for different ranges of water depth (i.e. depositional environments).
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    fig = plt.figure(figsize = (width,height))
    plt.rc("font", size=10)

    ax = fig.add_subplot(111)
    layID = []
    p = 0
    xi00 = cs.dist

    for i in range(0,cs.layNb+1,dnlay):
        if i == cs.layNb:
            i = cs.layNb-1
        layID.append(i)
        if len(layID) > 1:
            for j in range(0,len(xi00)-1):
                id = cs.secPropID[layID[p-1]][j]
                plt.fill_between([xi00[j],xi00[j+1]], [cs.secElev[layID[p-1]][j],
                                  cs.secElev[layID[p-1]][j+1]], color=color[id])
        p=p+1
    for i in range(0,cs.layNb,dnlay):
        if i>0:
            plt.plot(xi00,cs.secElev[i],'-',color='k',linewidth=0.2)
    plt.plot(xi00,cs.secElev[cs.layNb-1],'-',color='k',linewidth=0.7)
    plt.plot(xi00,cs.secElev[0],'-',color='k',linewidth=0.7)
    plt.xlim( rangeX )
    plt.ylim( rangeY )

    return

class sectionCarbonate:

    def __init__(self, folder=None, step=0, erange=None):
        """
        Initialization function which takes the folder path to Badlands outputs
        and the number of CPUs used to run the simulation.
        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        variable: step
            Step to output.
        variable: erange
            Extent of the box where computation is performed
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.step = step
        self.h5TIN = 'h5/tin.time'
        self.h5Strat = 'h5/stratal.time'
        self.erange = erange
        self.secTh = []
        self.secDep = []
        self.secElev = []
        self.secPropID = []

        self._build_dataset()

        return

    def _loadTIN(self, step):
        """
        Load TIN grid to extract cells connectivity and vertices position.
        Parameters
        ----------
        variable : step
            Specific step at which the TIN variables will be read.
        """

        h5file = self.folder+'/'+self.h5TIN+str(step)+'.hdf5'
        df = h5py.File(h5file, 'r')
        coords = np.array((df['/coords']))
        cells = np.array((df['/cells']),dtype=int)

        x, y, z = np.hsplit(coords, 3)
        dx = (x[1]-x[0])[0]
        nx = int((x.max() - x.min())/dx+1)
        ny = int((y.max() - y.min())/dx+1)

        if self.erange is None:
            xi = np.linspace(x.min(), x.max(), nx)
            yi = np.linspace(y.min(), y.max(), ny)
            xi, yi = np.meshgrid(xi, yi)
            xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
            XY = np.column_stack((x,y))
            tree = cKDTree(XY)

            distances, indices = tree.query(xyi, k=3)
            z_vals = z[indices][:,:,0]
            zi = np.average(z_vals,weights=(1./distances), axis=1)

            onIDs = np.where(distances[:,0] == 0)[0]
            if len(onIDs) > 0:
                zi[onIDs] = z[indices[onIDs,0],0]

            self.RegXi = xi
            self.RegYi = yi
            self.RegZi = np.reshape(zi,(ny,nx))
            self.RegExtent = [np.amin(xi), np.amax(xi), np.amin(yi), np.amax(yi)]
            self.inRange = np.arange(0,len(x))
        else:
            nnx = int((self.erange[1]-self.erange[0])/dx)+1
            nny = int((self.erange[3]-self.erange[2])/dx)+1
            xi = np.linspace(self.erange[0], self.erange[1], nnx)
            yi = np.linspace(self.erange[2], self.erange[3], nny)
            xi, yi = np.meshgrid(xi, yi)
            xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
            XY = np.column_stack((x,y))
            tree = cKDTree(XY)

            distances, indices = tree.query(xyi, k=3)
            z_vals = z[indices][:,:,0]
            zi = np.average(z_vals,weights=(1./distances), axis=1)

            onIDs = np.where(distances[:,0] == 0)[0]
            if len(onIDs) > 0:
                zi[onIDs] = z[indices[onIDs,0],0]

            self.RegXi = xi
            self.RegYi = yi
            self.RegZi = np.reshape(zi,(nny,nnx))
            self.RegExtent = [np.amin(xi), np.amax(xi), np.amin(yi), np.amax(yi)]
            inX = np.where(np.logical_and(x>=self.erange[0],x<=self.erange[1]))[0]
            inY = np.where(np.logical_and(y>=self.erange[2],y<=self.erange[3]))[0]
            self.inRange = np.intersect1d(inX, inY)

        return coords, cells

    def plotSectionMap(self, title='Section', color=None,  crange=None,
                                erange=None, pt=None, ctr='k',size=(8,8)):
        """
        Plot a given set of sections on the map

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

        variable: pt
            List of section points

        variable: size
            Figure size

        """

        rcParams['figure.figsize'] = size
        ax=plt.gca()

        if erange is not None:
            r1,c1 = np.where(np.logical_and(self.RegXi>=erange[0],self.RegXi<=erange[1]))
            r2,c2 = np.where(np.logical_and(self.RegYi>=erange[2],self.RegYi<=erange[3]))

            rdata = self.RegZi[r2.min():r2.max(),c1.min():c1.max()]

            im = ax.imshow(np.flipud(rdata),interpolation='nearest',cmap=color,
                       vmin=crange[0], vmax=crange[1], extent=erange)


            plt.contour(self.RegXi[r2.min():r2.max(),c1.min():c1.max()], self.RegYi[r2.min():r2.max(),c1.min():c1.max()],
                        self.RegZi[r2.min():r2.max(),c1.min():c1.max()], (0,), colors=ctr, linewidths=2)

        else:
            im = ax.imshow(np.flipud(self.RegZi),interpolation='nearest',cmap=color,
                           vmin=crange[0], vmax=crange[1], extent=self.RegExtent)

            plt.contour(self.RegXi, self.RegYi, self.RegZi, (0,), colors=ctr, linewidths=2)

        if pt is not None:
            for k in range(len(pt)):
                plt.plot(pt[k][:,0],pt[k][:,1], '-x', markersize=4)
        plt.title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.colorbar(im,cax=cax)
        plt.show()
        plt.close()

        return

    def _loadStrati(self, step):
        """
        Load stratigraphic dataset.
        Parameters
        ----------
        variable : step
            Specific step at which the TIN variables will be read.
        """

        h5file = self.folder+'/'+self.h5Strat+str(step)+'.hdf5'
        df = h5py.File(h5file, 'r')
        rockNb = len(df.keys())-1
        paleoH = np.array((df['/paleoDepth']))
        rockProp = np.zeros((paleoH.shape[0],paleoH.shape[1],rockNb))

        for r in range(rockNb):
            rockProp[:,:,r] = np.array((df['/depoThickRock'+str(r)]))

        return rockProp, paleoH

    def _build_dataset(self):

        s = self.step

        # Load TIN grid for specific time step
        coords, cells = self._loadTIN(s)
        x, y, z = np.hsplit(coords, 3)

        # Load STRATI dataset
        rockTH, paleoH = self._loadStrati(s)
        tmpRock = rockTH[self.inRange,1:]
        tmpPaleo = paleoH[self.inRange,1:]

        # Define dimensions
        ptsNb = len(self.inRange)
        layNb = paleoH.shape[1]
        rockNb = rockTH.shape[2]
        self.rockNb = rockNb
        self.layNb = layNb

        # Build attributes:
        # Layer number attribute
        layID = np.array([np.arange(layNb),]*ptsNb,dtype=int)
        ltmp = layID.flatten(order='F')

        # Thickness of each layer
        layTH = np.sum(tmpRock,axis=2)
        htmp = layTH.flatten(order='F')
        htmp = np.concatenate((np.zeros(ptsNb),htmp), axis=0)

        # Paleo-depth of each layer
        dtmp = tmpPaleo.flatten(order='F')
        dtmp = np.concatenate((dtmp,z[self.inRange,0]), axis=0)

        # Elevation of each layer
        cumH = np.cumsum(layTH[:,::-1],axis=1)[:,::-1]
        layZ = z[self.inRange] - cumH

        # Add the top surface to the layer elevation record
        ztmp = layZ.flatten(order='F')
        ztmp = np.concatenate((ztmp, z[self.inRange,0]), axis=0)

        # Proportion of each rock type
        r,c = np.where(layTH>0)

        rockProp = np.zeros((tmpRock.shape))
        for k in range(rockNb):
            rockProp[r,c,k] = np.divide(tmpRock[r,c,k],layTH[r,c])

        for cc in range(1,layNb-1):
            tt = np.where(layTH[:,cc]<=0)
            for k in range(rockNb):
                rockProp[tt,cc,k] = rockProp[tt,cc-1,k]

        ptmp = rockProp.reshape((ptsNb*(layNb-1),rockNb),order='F')
        ptmp = np.concatenate((np.zeros((ptsNb,rockNb)),ptmp), axis=0)

        self.dx = x[1]-x[0]
        self.nx = int((x[self.inRange].max() - x[self.inRange].min())/self.dx+1)
        self.ny = int((y[self.inRange].max() - y[self.inRange].min())/self.dx+1)
        self.x = x[self.inRange,0]
        self.y = y[self.inRange,0]
        self.z = ztmp.reshape((len(self.x),self.layNb),order='F')
        self.th = htmp.reshape((len(self.x),self.layNb),order='F')
        self.de = dtmp.reshape((len(self.x),self.layNb),order='F')
        self.propRock = []
        for k in range(self.rockNb):
            self.propRock.append(ptmp[:,k].reshape((len(self.x),self.layNb),order='F'))

        return

    def interpolate(self, dump=False):


        self.xi = np.linspace(self.x.min(), self.x.max(), self.nx)
        self.yi = np.linspace(self.y.min(), self.y.max(), self.ny)

        self.xx, self.yy = np.meshgrid(self.xi, self.yi)
        xyi = np.dstack([self.xx.flatten(), self.yy.flatten()])[0]
        XY = np.column_stack((self.x,self.y))
        tree = cKDTree(XY)

        # df = pd.DataFrame({'dX':self.x,'dY':self.y,'dZ':self.z[:,self.layNb-1]})
        # df.to_csv('elev.csv',columns=['dX', 'dY', 'dZ'], sep=',', index=False ,header=1)

        distances, indices = tree.query(xyi, k=3)
        offIDs = np.where(distances[:,0] > 0)[0]
        onIDs = np.where(distances[:,0] == 0)[0]

        self.rdep = np.zeros((self.ny,self.nx,self.layNb))
        self.relev = np.zeros((self.ny,self.nx,self.layNb))
        self.rth = np.zeros((self.ny,self.nx,self.layNb))
        self.rprop = np.zeros((self.ny,self.nx,self.layNb,self.rockNb))

        for k in range(1,self.layNb):
            tmp1 = self.z[:,k]
            z_vals = tmp1[indices]
            tmp2 = self.th[:,k]
            th_vals = tmp2[indices]
            tmp3 = self.de[:,k]
            de_vals = tmp3[indices]

            zi = np.zeros(len(xyi))
            hi = np.zeros(len(xyi))
            di = np.zeros(len(xyi))
            propi = np.zeros((len(xyi),self.rockNb))
            if len(offIDs) > 0:
                zi[offIDs] = np.average(z_vals[offIDs,:],weights=(1./distances[offIDs,:]), axis=1)
                di[offIDs] = np.average(de_vals[offIDs,:],weights=(1./distances[offIDs,:]), axis=1)
                hi[offIDs] = np.average(th_vals[offIDs,:],weights=(1./distances[offIDs,:]), axis=1)

            if len(onIDs) > 0:
                zi[onIDs] = tmp1[indices[onIDs,0]]
                di[onIDs] = tmp3[indices[onIDs,0]]
                hi[onIDs] = tmp2[indices[onIDs,0]]

            for s in range(self.rockNb):
                tmp4 = self.propRock[s][:,k]
                prop_vals = tmp4[indices]
                if len(offIDs) > 0:
                    propi[offIDs,s] = np.average(prop_vals[offIDs,:],weights=(1./distances[offIDs,:]), axis=1)

                if len(onIDs) > 0:
                    propi[onIDs,s] = tmp4[indices[onIDs,0]]

            self.rdep[:,:,k] = di.reshape((self.ny,self.nx))
            self.relev[:,:,k] = zi.reshape((self.ny,self.nx))
            self.rth[:,:,k] = hi.reshape((self.ny,self.nx))
            self.relev[:,:,0] = self.relev[:,:,1]
            for s in range(self.rockNb):
                self.rprop[:,:,k,s] = propi[:,s].reshape((self.ny,self.nx))

        if dump:
            self.rdep.dump("paleodep.dat")
            self.relev.dump("layelev.dat")
            self.rth.dump("layth.dat")
            self.rprop.dump("layProp.dat")

        return

    def _cross_section(self, xo, yo, xm, ym, pts):
        """
        Compute cross section coordinates.
        """

        if xm == xo:
            ysec = np.linspace(yo, ym, pts)
            xsec = np.zeros(pts)
            xsec.fill(xo)
        elif ym == yo:
            xsec = np.linspace(xo, xm, pts)
            ysec = np.zeros(pts)
            ysec.fill(yo)
        else:
            a = (ym-yo)/(xm-xo)
            b = yo - a * xo
            xsec = np.linspace(xo, xm, pts)
            ysec = a * xsec + b

        return xsec, ysec

    def buildSection(self, sec = None,
                    pts = None, gfilter = 0.001):
        """
        Extract a slice from the 3D data set and compute the stratigraphic layers.
        Parameters
        ----------
        variable: sec
            Section first and last point coordinates (X,Y).
        variable: pts
            Number of points to discretise the cross-section.
        variable: gfilter
            Gaussian smoothing filter.
        """

        if pts is None:
            pts = self.nx * 10

        xo = sec[0,0]
        xm = sec[1,0]

        yo = sec[0,1]
        ym = sec[1,1]

        if xm > self.x.max():
            xm = self.x.max()

        if ym > self.y.max():
            ym = self.y.max()

        if xo < self.x.min():
            xo = self.x.min()

        if yo < self.y.min():
            yo = self.y.min()

        xsec, ysec = self._cross_section(xo, yo, xm, ym, pts)
        self.dist = np.sqrt(( xsec - xo )**2 + ( ysec - yo )**2)
        self.xsec = xsec
        self.ysec = ysec
        for k in range(self.layNb):
            # Thick
            rect_B_spline = RectBivariateSpline(self.yi, self.xi, self.rth[:,:,k])
            data = rect_B_spline.ev(ysec, xsec)
            secTh = filters.gaussian_filter1d(data,sigma=gfilter)
            secTh[secTh < 0] = 0
            self.secTh.append(secTh)

            # Elev
            rect_B_spline1 = RectBivariateSpline(self.yi, self.xi, self.relev[:,:,k])
            data1 = rect_B_spline1.ev(ysec, xsec)
            secElev = filters.gaussian_filter1d(data1,sigma=gfilter)
            self.secElev.append(secElev)

            # Depth
            rect_B_spline2 = RectBivariateSpline(self.yi, self.xi, self.rdep[:,:,k])
            data2 = rect_B_spline2.ev(ysec, xsec)
            secDep = filters.gaussian_filter1d(data2,sigma=gfilter)
            self.secDep.append(secDep)

            # Prop
            idprop = np.zeros(secDep.shape[0],dtype=int)

            rect_B_spline3 = RectBivariateSpline(self.yi, self.xi, self.rprop[:,:,k,0])
            data3 = rect_B_spline3.ev(ysec, xsec)
            secProp1 = filters.gaussian_filter1d(data3,sigma=gfilter)

            rect_B_spline4 = RectBivariateSpline(self.yi, self.xi, self.rprop[:,:,k,1])
            data4 = rect_B_spline4.ev(ysec, xsec)
            secProp2 = filters.gaussian_filter1d(data4,sigma=gfilter)

            if self.rockNb>2:
                rect_B_spline5 = RectBivariateSpline(self.yi, self.xi, self.rprop[:,:,k,2])
                data5 = rect_B_spline5.ev(ysec, xsec)
                secProp3 = filters.gaussian_filter1d(data5,sigma=gfilter)
                r1 = np.where(np.logical_and(secProp2>secProp1,secProp2>secProp3))[0]
                idprop[r1] = 1
                r2 = np.where(np.logical_and(secProp3>secProp2,secProp3>secProp1))[0]
                idprop[r2] = 2
            else:
                r3 = np.where(secProp2>secProp1)[0]
                idprop[r3] = 1
            self.secPropID.append(idprop)

        # Ensure the spline interpolation does not create underlying layers above upper ones
        topsec = self.secDep[self.layNb-1]
        for k in range(self.layNb-2,-1,-1):
            secDep = self.secDep[k]
            self.secDep[k] = np.minimum(secDep, topsec)
            topsec = self.secDep[k]

        topsec = self.secElev[self.layNb-1]
        for k in range(self.layNb-2,-1,-1):
            secElev = self.secElev[k]
            self.secElev[k] = np.minimum(secElev, topsec)
            topsec = self.secElev[k]
            
        return
