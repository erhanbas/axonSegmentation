import numpy as np
import math
import re
from collections import defaultdict
import os
from skimage import io


def boundingbox(xyz):
    # finds the bounding box of point cloud
    return(np.round(np.array([np.min(xyz,axis=1),np.max(xyz,axis=1)])))

def snapLoc(I,nploc,w=1):
    # finds the location of maxima in a 3x3x3 neighborhood
    nplocupdated = nploc.copy()
    for iter,loc in enumerate(nploc):
        Icrop = I[loc[0]-w:loc[0]+w+1,
                  loc[1]-w:loc[1]+w+1,
                  loc[2]-w:loc[2]+w+1]
        i, j, k = np.unravel_index(Icrop.argmax(), Icrop.shape)
        nplocupdated[iter] = loc+(np.array((i,j,k))-w)
    return nplocupdated


def xyz2oct(xyz,params):
    # converts xyz location to oct location
    if len(xyz.shape) ==1:
        xyz=xyz[None,:]

    nlevel = np.int(params['nlevels'])
    leafsize = params['leafshape']
    octpath = np.zeros((xyz.shape[0],nlevel))
    xres = np.zeros((xyz.shape[0],3))
    for idx in range(xyz.shape[0]):
        bits = []
        x = xyz[idx]
        u = leafsize
        for n in range(nlevel-1,-1,-1):
            bn = 2**n*u
            th = x>bn
            bits.append(th)
            x = x - bn*th
        # convert to octodigit
        octpath[idx,:] = (1+ np.sum(np.array(bits)*2**np.array([0,1,2]),axis=1))[None,:]
        xres[idx,:] = x

    return octpath.astype(int),xres

def to_base_3(n):
    s = ""
    while n:
        print(n)
        s = str(n % 3) + s
        n = round(n/3)
    return s

def to_base_2(n,numdigit=0):
    n = math.floor(n)
    s = ""
    while n:
        s = str(n % 2) + s
        n = math.floor(n/2)
    s=(numdigit-len(s))*'0'+s
    return s

def oct2grid_list(octpath):
    depth = octpath.shape[1]
    numpath = octpath.shape[0]
    alltiles = []
    for ijk in octpath:
        xyz = oct2grid(ijk.reshape(1,depth))
        # for every path, there are 26 neighbors
        alltiles.append(xyz[None,:])
    if len(alltiles) == 1:
        alltiles = np.squeeze(alltiles[0])
    else:
        alltiles = np.squeeze(np.concatenate(alltiles, axis=1))
    return alltiles

def oct2grid(oct_idx):
    # (inverse logic as grid2oct)
    # oct_idx [1..8]
    # grid [0 dims]
    if np.any(oct_idx < 1) or np.any(oct_idx > 8):
        raise Exception('oct out of bound')

    if oct_idx.ndim == 1:
        oct_idx = oct_idx.reshape(1,len(oct_idx))

    numlist = oct_idx.shape[0]
    depth = oct_idx.shape[1]
    binarray = 2 ** (np.array(range(depth, 0, -1)) - 1)
    gridarray = np.zeros((numlist,3))
    for il in range(numlist):
        idxarray = np.zeros((3, depth))
        for id in range(depth):
            base2 = to_base_2(oct_idx[il,id]-1, 3)
            idxarray[2, id] = int(base2[0])
            idxarray[1, id] = int(base2[1])
            idxarray[0, id] = int(base2[2])
        gridarray[il,:] = np.sum(idxarray * binarray, axis=1)
    # broadcast binarray
    return(np.asarray(gridarray,dtype=int))

def loadTiles(tilepath,ext=".tif"):
    IM=[]
    files = os.listdir(tilepath)
    files.sort() # make sure that channels are loaded in order
    for file in files:
        if file.endswith(ext):
            tilefiles = os.path.join(tilepath, file)
            # load tile
            im = io.imread(tilefiles)  # zyx order
            IM.append(np.swapaxes(im, 0, 2))

    return np.stack(IM,axis=3)


def grid2oct(xyz,depth):
    # order flip to pre (inverse logic as oct2grid)
    numlist = xyz.shape[0]
    outijk = np.zeros((numlist, depth), dtype=np.int)
    for il in range(numlist):
        arr = []
        arr.append(to_base_2(xyz[il,2], depth))
        arr.append(to_base_2(xyz[il,1], depth))
        arr.append(to_base_2(xyz[il,0], depth))
        for idx in range(depth):
            b = [el[idx] for el in arr]
            outijk[il,idx]=np.int(''.join(b),2)
    return(outijk+1)

def chunklist(pathlist,depth):
    # -> for each tile, find bbox of crop sub-octtree
    listdict = defaultdict(list)
    for tileid in pathlist:
        # list of crops for this tile
        mykey = re.sub('[\[\]]', '', np.array_str(tileid).replace(' ',''))
        listdict[mykey[:depth]].append(mykey[depth:])

    return listdict

def dilateOct(octpath,width=1):
    # dilates the octpath with the given search widty
    # 1/2/3 with width 1 -> 1/2/3 | 2/2/3 | 1/1/3 | 1/3/3 | ... | 2/3/4
    depth = octpath.shape[1]
    numpath = octpath.shape[0]
    ix, iy, iz = np.mgrid[-width:width+1, -width:width+1, -width:width+1]
    ixyz = np.stack((ix.flatten(), iy.flatten(), iz.flatten()), axis=1)
    alltiles = []
    for ijk in octpath:
        xyz = oct2grid(ijk.reshape(1,depth))
        # for every path, there are 26 neighbors
        alltiles.append(xyz[None,:]+ixyz)

    if len(alltiles) == 1:
        alltiles = np.squeeze(alltiles[0])
    else:
        alltiles = np.squeeze(np.concatenate(alltiles, axis=1))
    # delete any out of bound tiles
    deletethese = np.any(np.logical_or(alltiles < 0, alltiles > 2**depth-1), axis=1)
    alltiles = np.delete(alltiles,(np.where(deletethese)),axis=0)

    # unique entries
    alltiles = np.unique(alltiles,axis=0)
    # convert to octpaths
    octlist = [grid2oct(tileid[None,:], depth) for tileid in alltiles]
    octlist = np.concatenate(octlist, axis=0)

    return octlist,alltiles

# def boundingboxOctree(xyz,params):
#     # finds the bounding box of point cloud wrto octree
#
#     x = xyz[]
#     for idx in range(nlevel,0,-1):
#         th = xyz
#
#
#     return(np.round(np.array([np.min(xyz,axis=1),np.max(xyz,axis=1)])))
