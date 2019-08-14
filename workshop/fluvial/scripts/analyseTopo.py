##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to analyse topographic changes from Badlands outputs.
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

def getDisplayInterval(folder=None):

    filename = glob.glob(folder+'/*.xml')
    with open(filename[0]) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if 'display' in line:
                val = [float(s) for s in re.findall(r'-?\d+\.?\d*', line)]
                tdisplay = val[0]
                line = False
            else:
                line = fp.readline()
            cnt += 1

    return tdisplay

def loadDataTIN(folder=None, timestep=0):

    if not os.path.isdir(folder):
        raise RuntimeError('The given folder cannot be found or the path is incomplete.')

    df = h5py.File('%s/tin.time%s.hdf5'%(folder, timestep), 'r')
    coords = np.array((df['/coords']))
    cumdiff = np.array((df['/cumdiff']))
    cumhill = np.array((df['/cumhill']))
    discharge = np.array((df['/discharge']))

    return coords, cumdiff

def readDataset(folder=None, isldPos=None, isldRadius=None, refPos=None, pltRadius=None):

    stepCounter = len(glob.glob1(folder+"/xmf/","tin.time*"))
    steps = np.arange(0,stepCounter,1)

    tdisplay = getDisplayInterval(folder)
    time = steps*tdisplay

    # Erosion/deposition for island and reference and their corresponding plateaus
    i_ed = np.zeros((len(steps),3))
    r_ed = np.zeros((len(steps),3))
    pi_ed = np.zeros((len(steps),3))
    pr_ed = np.zeros((len(steps),3))

    # Elevation for island and reference and their corresponding plateaus
    i_z = np.zeros((len(steps),5))
    r_z = np.zeros((len(steps),3))
    pi_z = np.zeros((len(steps),3))
    pr_z = np.zeros((len(steps),3))

    for k in range(len(steps)):
        if k%50 == 0:
            print('Time step:',steps[k])

        nfile =folder
        xyz, ed = loadDataTIN(nfile+'/h5',steps[k])
        x, y, z = np.hsplit(xyz, 3)
        if k == 0:
            tree = spatial.cKDTree(xyz[:,:2])
            # Big island region
            isldXY = isldPos
            dist = isldRadius
            islds = tree.query_ball_point(isldXY,dist)
            # Ref region circle
            refXY = refPos
            refs = tree.query_ball_point(refXY,dist)
            # Plateau facing island
            tmp1 = tree.query_ball_point(isldXY,pltRadius*2.)
            a1 = np.asarray(islds)
            b1 = np.asarray(tmp1)
            plat1 = np.setdiff1d(b1,a1)
            # Plateau 2
            tmp2 = tree.query_ball_point(refXY,pltRadius*2.)
            a2 = np.asarray(refs)
            b2 = np.asarray(tmp2)
            plat2 = np.setdiff1d(b2,a2)

        # Mean erosion/deposition
        i_ed[k,0] = np.min(ed[islds])
        r_ed[k,0] = np.min(ed[refs])
        pi_ed[k,0] = np.min(ed[plat1])
        pr_ed[k,0] = np.min(ed[plat2])

        i_ed[k,1] = np.mean(ed[islds])
        r_ed[k,1] = np.mean(ed[refs])
        pi_ed[k,1] = np.mean(ed[plat1])
        pr_ed[k,1] = np.mean(ed[plat2])

        i_ed[k,2] = np.max(ed[islds])
        r_ed[k,2] = np.max(ed[refs])
        pi_ed[k,2] = np.max(ed[plat1])
        pr_ed[k,2] = np.max(ed[plat2])

        i_z[k,1] = np.mean(z[islds])
        i_z[k,3] = np.percentile(z[islds], 5)
        i_z[k,4] = np.percentile(z[islds], 95)
        r_z[k,1] = np.mean(z[refs])
        pi_z[k,1] = np.mean(z[plat1])
        pr_z[k,1] = np.mean(z[plat2])

        i_z[k,0] = np.min(z[islds])
        r_z[k,0] = np.min(z[refs])
        pi_z[k,0] = np.min(z[plat1])
        pr_z[k,0] = np.min(z[plat2])

        i_z[k,2] = np.max(z[islds])
        r_z[k,2] = np.max(z[refs])
        pi_z[k,2] = np.max(z[plat1])
        pr_z[k,2] = np.max(z[plat2])

    # Define cumulative erosion dataset
    cumi = np.zeros((len(steps),3))
    cumr = np.zeros((len(steps),3))
    cumpi = np.zeros((len(steps),3))
    cumpr = np.zeros((len(steps),3))

    for k in range(1,len(steps)):
        for p in range(3):
            # Cumulative erosion
            cumi[k,p] = -i_ed[k,p]
            cumr[k,p] = -r_ed[k,p]
            cumpi[k,p] = -pi_ed[k,p]
            cumpr[k,p] = -pr_ed[k,p]

    islands = []
    islands.append(i_z)
    islands.append(pi_z)
    islands.append(cumi)
    islands.append(cumpi)

    plateaus = []
    plateaus.append(r_z)
    plateaus.append(pr_z)
    plateaus.append(cumr)
    plateaus.append(cumpr)

    return time,islands,plateaus

def elevationChange(title='Title',time=None,island=None,plateau=None,figsave=None):

    rcParams['figure.figsize'] = (9,6)
    rcParams['font.size'] = 8

    fig, (ax, ax1) = plt.subplots(1, 2, sharey=True)

    st = fig.suptitle(title, fontsize=10)

    ax.plot(time, island[0][:,1], color='#009933',linewidth='3',label='island ',zorder=3)
    ax.fill_between(time, island[0][:,0], island[0][:,2], where=island[0][:,2] > island[0][:,0], facecolor='#009933', alpha=0.25,zorder=1)

    ax.plot(time, plateau[0][:,1], color='#ff9900',linewidth='3',label='ref ',zorder=5)
    ax.fill_between(time, plateau[0][:,0], plateau[0][:,2], where=plateau[0][:,2] > plateau[0][:,0], facecolor='#ff9900', alpha=0.25,zorder=1)
    lgd = ax.legend(loc=1, fontsize=10)
    ax.grid(True)
    ax.set_ylabel('Elevation [m]', fontsize=11)
    ax.set_xlabel('Time [y]', fontsize=10)

    ax1.plot(time, island[1][:,1], color='#009933',linewidth='3',label='plateau island ',zorder=2)
    ax1.plot(time, plateau[1][:,1], color='#ff9900',linewidth='3',label='plateau ref ',zorder=2)
    ax1.fill_between(time, island[1][:,0], island[1][:,2], where=island[1][:,2] > island[1][:,0], facecolor='#009933', alpha=0.25,zorder=1)
    ax1.fill_between(time, plateau[1][:,0], plateau[1][:,2], where=plateau[1][:,2] > plateau[1][:,0], facecolor='#ff9900', alpha=0.25,zorder=1)

    lgd = ax1.legend(loc=1, fontsize=10)
    ax1.grid(True)
    ax1.set_xlabel('Time [y]', fontsize=10)
    st.set_y(1.02)
    fig.subplots_adjust(top=0.85)
    plt.show()

    if figsave is not None:
        fig.savefig(figsave)

    plt.close()

def cumulativeErosion(title='Title',time=None,island=None,plateau=None,figsave=None):

    rcParams['figure.figsize'] = (9,6)
    rcParams['font.size'] = 8

    fig, (ax, ax1) = plt.subplots(1, 2, sharey=True)

    st = fig.suptitle(title, fontsize=10)

    ax.plot(time, island[2][:,1], color='#009933',linewidth='3',label='island ',zorder=3)
    ax.fill_between(time, island[2][:,0], island[2][:,2], where=island[2][:,2] < island[2][:,0], facecolor='#009933', alpha=0.25,zorder=1)
    ax.plot(time, plateau[2][:,1], color='#ff9900',linewidth='3',label='ref ',zorder=5)
    ax.fill_between(time, plateau[2][:,0], plateau[2][:,2], where=plateau[2][:,2] < plateau[2][:,0], facecolor='#ff9900', alpha=0.25,zorder=1)
    lgd = ax.legend(loc=2, fontsize=10)
    ax.grid(True)
    ax.set_ylabel('Cumulative erosion [m]', fontsize=11)
    ax.set_xlabel('Time [y]', fontsize=10)

    ax1.plot(time, island[3][:,1], color='#009933',linewidth='3',label='plateau island ',zorder=2)
    ax1.plot(time, plateau[3][:,1], color='#ff9900',linewidth='3',label='plateau ref ',zorder=2)
    ax1.fill_between(time, plateau[3][:,0], plateau[3][:,2], where=plateau[3][:,2] < plateau[3][:,0], facecolor='#ff9900', alpha=0.25,zorder=1)
    ax1.fill_between(time, island[3][:,0], island[3][:,2], where=island[3][:,2] < island[3][:,0], facecolor='#009933', alpha=0.25,zorder=1)

    lgd = ax1.legend(loc=2, fontsize=10)
    ax1.grid(True)
    ax1.set_xlabel('Time [y]', fontsize=10)
    st.set_y(1.02)
    fig.subplots_adjust(top=0.85)
    plt.show()

    if figsave is not None:
        fig.savefig(figsave)

    plt.close()

def erosionRate(title='Title',time=None,island=None,plateau=None,figsave=None):

    rcParams['figure.figsize'] = (9,6)
    rcParams['font.size'] = 8

    fig, (ax, ax1) = plt.subplots(1, 2, sharey=True)

    st = fig.suptitle(title, fontsize=10)
    dt = time[1]-time[0]
    eromean = np.gradient(island[2][:,1], dt, edge_order=2)*1000.
    ax.plot(time, eromean, color='#009933',linewidth='3',label='island ',zorder=3)
    eromean = np.gradient(plateau[2][:,1], dt, edge_order=2)*1000.
    ax.plot(time, eromean, color='#ff9900',linewidth='3',label='ref ',zorder=5)

    lgd = ax.legend(loc=1, fontsize=10)
    ax.grid(True)
    ax.set_ylabel('Erosion rate [mm/y]', fontsize=11)
    ax.set_xlabel('Time [y]', fontsize=10)

    eromean = np.gradient(island[3][:,1], dt, edge_order=2)*1000.
    ax1.plot(time, eromean, color='#009933',linewidth='3',label='plateau island ',zorder=2)

    eromean = np.gradient(plateau[3][:,1], dt, edge_order=2)*1000.
    ax1.plot(time, eromean, color='#ff9900',linewidth='3',label='plateau ref ',zorder=2)

    lgd = ax1.legend(loc=1, fontsize=10)
    ax1.grid(True)
    ax1.set_xlabel('Time [y]', fontsize=10)
    st.set_y(1.02)
    fig.subplots_adjust(top=0.85)
    plt.show()

    if figsave is not None:
        fig.savefig(figsave)

    plt.close()
