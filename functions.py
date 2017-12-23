from morphsnakes import morphsnakes as smorph
# from chanvese import chanvese3d as cv3d
import numpy as np

def createBinaryMask(size,point1,point2):
    # creates a box around points
    it=1


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u

def boundingbox_levelset(shape, center, sqradius, scalerow=1.0):
    test=1

# def snake2(img_,init):
#     if img_.ndim>3:
#         img = img_[:,:,:,0]
#     else:
#         img = img_
#     init_mask = circle_levelset(img.shape, init, 10)
#     cv3d.chanvese3d(img, init_mask, max_its=200, alpha=0.2, thresh=0, color='r', display=True)

def snake(img_,init):
    # img = np.load("./morphsnakes/testimages/confocal.npy")
    # fig = plt.figure(frameon=False)
    if img_.ndim>3:
        img = img_[:,:,:,0]
    else:
        img = img_

    print(img.shape)
    if True:
        macwe = smorph.MorphACWE(img, smoothing=1, lambda1=1, lambda2=2)
        macwe.levelset = circle_levelset(img.shape, init, 10)
        macwe.levelset = boundingbox_levelset()
        smorph.evolve_visual3d(macwe, num_iters=100)
    else:
        # g(I)
        gI = smorph.gborders(img, alpha=1000, sigma=2)
        # Morphological GAC. Initialization of the level-set.
        mgac = smorph.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-1)
        mgac.levelset = circle_levelset(img.shape, init, 10)
        smorph.evolve_visual3d(mgac, num_iters=100)

# `vol` your already segmented 3d-lungs, using one of the other scripts
# `mask` you can start with all 1s, and after this operation, it'll have 0's where you need to delete
# `start_point` a tuple of ints with (z, y, x) coordinates
# `epsilon` the maximum delta of conductivity between two voxels for selection
# `HU_mid` Hounsfield unit midpoint
# `HU_range` maximim distance from `HU_mid` that will be accepted for conductivity
# `fill_with` value to set in `mask` for the appropriate location in vol that needs to be flood filled

def region_grow(vol, mask, start_point, epsilon=5, HU_mid=0, HU_range=0, fill_with=1):
    sizex = vol.shape[0] - 1
    sizey = vol.shape[1] - 1
    sizez = vol.shape[2] - 1

    items = []
    visited = []

    def enqueue(item):
        items.insert(0, item)

    def dequeue():
        s = items.pop()
        visited.append(s)
        return s

    print(start_point.shape)
    enqueue((start_point[0], start_point[1], start_point[2]))
    beta=0.99
    updatemean = 1

    while not items == []:

        x, y, z = dequeue()
        if x==6 and y == 1 and z==2:
            vizneig=1
        else:
            vizneig = 0

        voxel = vol[x, y, z]
        mask[x, y, z] = fill_with
        print(x, y, z, voxel, HU_mid,len(items))
        if x < sizex and mask[x+1, y, z] !=fill_with:
            tvoxel = vol[x+1, y, z]
            if vizneig:print("+x",x+1, y, z, voxel, tvoxel, HU_mid)
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((x+1, y, z));
                if updatemean:HU_mid=beta*HU_mid+(1-beta)*tvoxel

        if x >0 and mask[x-1, y, z] !=fill_with:
            tvoxel = vol[x-1, y, z]
            if vizneig:print("-x",x-1, y, z, voxel, tvoxel, HU_mid)
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((x-1, y, z));
                if updatemean: HU_mid = beta * HU_mid + (1 - beta) * tvoxel

        if y < sizey and mask[x, y+1, z] !=fill_with:
            tvoxel = vol[x, y+1, z]
            if vizneig:print("+y",x, y+1, z, voxel, tvoxel, HU_mid)
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((x,y+1, z));
                if updatemean: HU_mid = beta * HU_mid + (1 - beta) * tvoxel

        if y >0 and mask[x, y-1, z] !=fill_with:
            tvoxel = vol[x, y-1, z]
            if vizneig:print("-y",x, y-1, z, voxel, tvoxel, HU_mid)
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((x, y-1, z));
                if updatemean:HU_mid=beta*HU_mid+(1-beta)*tvoxel

        if z < sizez and mask[x, y, z+1] !=fill_with:
            tvoxel = vol[x, y, z+1]
            if vizneig:print("+z",x, y, z+1, voxel, tvoxel, HU_mid)
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((x, y, z+1));
                if updatemean: HU_mid = beta * HU_mid + (1 - beta) * tvoxel

        if z >0 and mask[x, y, z-1] !=fill_with:
            tvoxel = vol[x, y, z-1]
            if vizneig:print("-z",x, y, z-1, voxel, tvoxel, HU_mid)
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((x, y, z-1));
                if updatemean: HU_mid = beta * HU_mid + (1 - beta) * tvoxel

        # print(x, y, z, voxel, tvoxel, HU_mid)
    return mask
