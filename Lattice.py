#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:57 2021

@author: B.Chen @ HKU
"""
import numpy as np
import scipy.io as spio
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Basis import Basis

def get_dist(XY1, XY2):
    return np.sqrt((XY1-XY2)@(XY1-XY2).transpose())

class Lattice:
    def __init__(self, name='Square', length=12, width=4, boundary_x='OBC', boundary_y='PBC'):
        self.name = name
        self.length = length
        self.width = width
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.num_site = self.length * self.width
        self.siteorder_1d = self.get_siteorder_1d()
        self.siteorder_2d = self.get_siteorder_2d()
        self.coordinate, self.trans_vector = self.get_coordinate()
    
    def get_siteorder_1d(self):
        """
        obtain the 1d zig-zag order of a 2d lattice:
          \col| 0  1  2 ...
        row\  |--------------
        0     | 0  4  8  ...
        1     | 1  5  9  ...
        2     | 2  6  10  ...
        3     | 3  7  11  ...
        ---------------------
        !!! N.B. input (#row, #col) return a site order #site
            e.g. (3,1) --> 7
        """
        W = self.width
        L = self.length
        N = self.num_site
        return np.reshape(np.arange(0,N),[L,W]).transpose()

    def get_siteorder_2d(self):
        """
        obtain the 2d coordinate of the 1d zig-zag order:
          \col| 0  1  2 ...
        row\  |--------------
        0     | 0  4  8  ...
        1     | 1  5  9  ...
        2     | 2  6  10  ...
        3     | 3  7  11  ...
        ---------------------
        !!! N.B., input (#site) return a 2d vector (#row, #col)
            e.g.  7 --> (3,1) 
        """
        N = self.num_site
        W = self.width
        return [(si%W, si//W) for si in range(N)]

    def get_coordinate(self):
        """
        Given the type of lattice
        obtain the real-space coordinate of each site
        return a 2d vector (#x, #y)
        !!! Note the difference with siteorder_2d 
                     (exchanging row with col)
        """
        W = self.width
        L = self.length
        Basis_ = next(item for item in Basis if item["name"] == self.name)
        unit_cell = Basis_['unit_cell']
        prim_vec = np.array(Basis_['primitive_vector'])

        # checking the width
        num_site_unit = len(unit_cell)
        if W%num_site_unit:
            raise Exception('Error: width of Honeycomb_YC must be multiples of %d'%num_site_unit)
        else:
            W = W//num_site_unit

        # generating coordinates
        coordinate = []
        for ix in range(L):
            for iy in range(W):
                for xy_ in unit_cell:
                    xy_ = np.array(xy_)
                    coordinate.append(xy_ + ix*prim_vec[0] + iy*prim_vec[1])
        trans_vector = [L*prim_vec[0], W*prim_vec[1]]
        return coordinate, trans_vector

    def get_coordinate_bak(self):
        """
        Given the type of lattice
        obtain the real-space coordinate of each site
        return a 2d vector (#x, #y)
        !!! Note the difference with siteorder_2d 
                     (exchanging row with col)
        """
        W = self.width
        L = self.length
        # =================================================================
        if self.name == 'Square':
            coordinate = [Ord2d[::-1] for Ord2d in self.siteorder_2d ]
            # as the PBC information missing, we must specify where 
            # every site end up with by adding the trans_vector
            trans_vector = [(L,0), (0,W)]
            return coordinate, trans_vector
        # =================================================================
        elif self.name == 'Triangle_YC':
            coordinate = [(Ord2d[1]*np.sqrt(3)/2, Ord2d[0]+Ord2d[1]*0.5) for Ord2d in self.siteorder_2d]
            # as the PBC information missing, we must specify where 
            # every site end up with by adding the trans_vector
            trans_vector = [(np.sqrt(3)/2*L,0.5*L), (0,W)]
            return coordinate, trans_vector
        # =================================================================
        elif self.name == 'Honeycomb_YC':
            if self.width%2:
                raise Exception('Error: width of Honeycomb_YC must be even')
            coordinate = []
            for ord2d in self.siteorder_2d:
                row_ = ord2d[0]
                col_ = ord2d[1]
                xx_ = col_ * np.sqrt(3)/2
                yy_ = (row_+row_%2)/2*3 -(row_%2)*(2-col_%2) -(col_%2)/2
                coordinate.append((xx_, yy_))
            # as the PBC information missing, we must specify where 
            # every site end up with by adding the trans_vector
            trans_vector = [(np.sqrt(3)/2*L,0),
                            (0,W/2*3)]
            return coordinate, trans_vector
        # =================================================================
        elif self.name == 'Kagome_YC':
            if self.width%3: 
                raise Exception('Error: width must be multiples of 3 in Kagome_YC')
            coordinate = []
            for ord2d in self.siteorder_2d:
                row_ = ord2d[0]
                col_ = ord2d[1]
                xx_ = col_ * np.sqrt(3) - (row_%3==1)*np.sqrt(3)/2
                yy_ = (row_-row_%3)*2/3 + (row_%3)*0.5 - col_
                coordinate.append((xx_, yy_))
            # as the PBC information missing, we must specify where 
            # every site end up with by adding the trans_vector
            trans_vector = [(np.sqrt(3)*L,-L),
                            (0,W/3*2)]
            return coordinate, trans_vector
        # =================================================================
        else:
            return None, None


    def get_nearestneighbor(self):
        """
        obtain all nearest neighboring pairs of sites 
        according to the self.coordinates
        """
        XY = np.array(self.coordinate)
        N = self.num_site
        Trans = np.array(self.trans_vector)

        NNPairs = []
        for si in range(N):
            for sj in range(si+1,N):
                Dist_ = get_dist(XY[si,:], XY[sj,:])
                if np.abs(Dist_-1)<1e-13: 
                    NNPairs.append((si,sj,0,0))
                if self.boundary_x == 'PBC':
                    Dist_ = get_dist(XY[si,:], XY[sj,:]+Trans[0,:])
                    if np.abs(Dist_-1)<1e-13: 
                        NNPairs.append((si,sj,1,0))
                    Dist_ = get_dist(XY[si,:], XY[sj,:]-Trans[0,:])
                    if np.abs(Dist_-1)<1e-13: 
                        NNPairs.append((si,sj,-1,0))
                if self.boundary_y == 'PBC':
                    Dist_ = get_dist(XY[si,:], XY[sj,:]+Trans[1,:])
                    if np.abs(Dist_-1)<1e-13: 
                        NNPairs.append((si,sj,0,1))
                    Dist_ = get_dist(XY[si,:], XY[sj,:]-Trans[1,:])
                    if np.abs(Dist_-1)<1e-13: 
                        NNPairs.append((si,sj,0,-1))
        return NNPairs
    
    def get_bond(self, distance=1):
        """
        obtain all pairs of sites with distance
        according to the self.coordinates
        """
        XY = np.array(self.coordinate)
        N = self.num_site
        Trans = np.array(self.trans_vector)

        tol = 1e-13
        Pairs = []
        for si in range(N):
            for sj in range(si+1,N):
                Dist_ = get_dist(XY[si,:], XY[sj,:])
                if np.abs(Dist_-distance)<tol: 
                    Pairs.append((si,sj,0,0))
                if self.boundary_x == 'PBC':
                    Dist_ = get_dist(XY[si,:], XY[sj,:]+Trans[0,:])
                    if np.abs(Dist_-distance)<tol: 
                        Pairs.append((si,sj,1,0))
                    Dist_ = get_dist(XY[si,:], XY[sj,:]-Trans[0,:])
                    if np.abs(Dist_-distance)<tol: 
                        Pairs.append((si,sj,-1,0))
                if self.boundary_y == 'PBC':
                    Dist_ = get_dist(XY[si,:], XY[sj,:]+Trans[1,:])
                    if np.abs(Dist_-distance)<tol: 
                        Pairs.append((si,sj,0,1))
                    Dist_ = get_dist(XY[si,:], XY[sj,:]-Trans[1,:])
                    if np.abs(Dist_-distance)<tol: 
                        Pairs.append((si,sj,0,-1))
        return Pairs
    

    def plot_latt(self):
        """
        plot the lattice sites
        """
        N = self.num_site
        XY = np.array(self.coordinate)
        dist_ = get_dist(self.coordinate[0],self.coordinate[1])
        NNPairs = self.get_bond(distance=dist_)
        TransVector = np.array(self.trans_vector)
        TransX = TransVector[0,:]
        TransY = TransVector[1,:]
        
        LightBlue = [.6,.6,.9]
        ax = plt.axes()
        ax.plot(XY[:,0], XY[:,1], 'ko', zorder=100)

        for si in range(N):
            ax.text(XY[si,0], XY[si,1], r'%d'%si, transform=ax.transData, color='r',zorder=150,fontsize=6)

        if self.boundary_x == 'PBC':
            ax.plot(XY[:,0]+TransX[0], 
                    XY[:,1]+TransX[1], 'o', color='lightgrey', zorder=100)
            ax.plot(XY[:,0]-TransX[0], 
                    XY[:,1]-TransX[1], 'o', color='lightgrey', zorder=100)
        if self.boundary_y == 'PBC':
            ax.plot(XY[:,0]+TransY[0], 
                    XY[:,1]+TransY[1], 'o', color='lightgrey', zorder=100)
            ax.plot(XY[:,0]-TransY[0], 
                    XY[:,1]-TransY[1], 'o', color='lightgrey', zorder=100)

        for nnpairinfo in NNPairs:
            nnpair = nnpairinfo[:2]
            info = nnpairinfo[2:]

            ax.plot(XY[nnpair,0]+[0,TransX[0]*info[0]+TransY[0]*info[1]], 
                    XY[nnpair,1]+[0,TransX[1]*info[0]+TransY[1]*info[1]], 'b-')
            if self.boundary_x == 'PBC':
                ax.plot(XY[nnpair,0]+TransX[0]+[0,TransX[0]*info[0]+TransY[0]*info[1]], 
                        XY[nnpair,1]+TransX[1]+[0,TransX[1]*info[0]+TransY[1]*info[1]], 
                        '-', color=LightBlue)
                ax.plot(XY[nnpair,0]-TransX[0]+[0,TransX[0]*info[0]+TransY[0]*info[1]], 
                        XY[nnpair,1]-TransX[1]+[0,TransX[1]*info[0]+TransY[1]*info[1]], 
                        '-', color=LightBlue)
            if self.boundary_y == 'PBC':
                ax.plot(XY[nnpair,0]+TransY[0]+[0,TransX[0]*info[0]+TransY[0]*info[1]], 
                        XY[nnpair,1]+TransY[1]+[0,TransX[1]*info[0]+TransY[1]*info[1]], 
                        '-', color=LightBlue)
                ax.plot(XY[nnpair,0]-TransY[0]+[0,TransX[0]*info[0]+TransY[0]*info[1]], 
                        XY[nnpair,1]-TransY[1]+[0,TransX[1]*info[0]+TransY[1]*info[1]], 
                        '-', color=LightBlue)

        ax.axis('equal')
        plt.show()


if __name__=="__main__":
    latt1 = Lattice(name='Kagome_YC', width=6, boundary_y='PBC')
    latt1.plot_latt()
    
