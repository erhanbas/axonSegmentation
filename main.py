# import util
import numpy as np
# import pyface.qt
import matplotlib.pyplot as plt
# import improc
# import os
# from scipy.spatial import distance as dist
import h5py
from frangi3d import frangi
# from morphsnakes import tests
# from morphsnakes import morphsnakes as morph
import importlib
import functions as func

if __name__ == '__main__':
    # this is the main
    h5file = "./data/neuron_test.hdf5"
    # with h5py.File(h5file, "r") as f:
    f = h5py.File(h5file, "r")
    recon = f["reconstruction"]
    volume = f["volume"]
##
    ix=1
    start = recon[ix,2:5]
    end = recon[ix+1,2:5]
    start_ = np.asarray(start,np.int)
    end_ = np.asarray(end,np.int)
##
    shift = 0
    bbox_min = np.min((start_,end_),axis=0)-shift
    bbox_max = np.max((start_,end_),axis=0)+shift+1
    bbox_size = bbox_max-bbox_min
    start_ = start_ - bbox_min
    end_ = end_-bbox_min

    # crop
    crop = volume[bbox_min[0]:bbox_max[0],bbox_min[1]:bbox_max[1],bbox_min[2]:bbox_max[2]]
    # swap x-z for visualization
    if False:
        crop = crop.swapaxes(0,2)
        bbox_size = bbox_size[[2,1,0]]
        start_ = start_[[2,1,0]]
        end_ = end_[[2,1,0]]

    # plt.imshow(np.max(crop[:,:,:,0], axis=0), cmap='gray')
##
    # `vol` your already segmented 3d-lungs, using one of the other scripts
    # `mask` you can start with all 1s, and after this operation, it'll have 0's where you need to delete
    # `start_point` a tuple of ints with (z, y, x) coordinates
    # `epsilon` the maximum delta of conductivity between two voxels for selection
    # `HU_mid` Hounsfield unit midpoint
    # `HU_range` maximim distance from `HU_mid` that will be accepted for conductivity
    # `fill_with` value to set in `mask` for the appropriate location in vol that needs to be flood filled
    inputim = np.asarray(crop[:,:,:,0], np.float)
##
    # shorthestpath between two points

    importlib.reload(frangi)
    scale_range = (1, 10);scale_step = 2
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    sd,ds=frangi.frangi(inputim, scale_range, scale_step, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=False)


##
    mask = inputim*0
    start_point = start_
    start_point = np.array([7,1,3])
    import functions as func
    reload(func)
    seg = func.region_grow(inputim, mask, start_point, epsilon=1000, HU_mid=inputim[start_point[0],start_point[2],start_point[2]], HU_range=1000, fill_with=255)

    plt.figure()
    for ii in range(5):
        ax=plt.subplot(1,5,ii+1)
        ax.imshow(inputim[:,:,ii].T, cmap='gray', interpolation='nearest')
        ax.imshow(mask[:, :, ii].T, cmap='jet', interpolation='nearest', alpha=.3)
        ax.set_title(ii)
        # ax.tick_params(labeltop=True, labelright=True)
##
#         # plt.figure()
#     # plt.imshow(np.max(seg, axis=2).T, cmap='gray')
#
# ##
#     import importlib
#     import functions as func
#     reload(func)
#     func.snake2(crop**(0.5),bbox_size/2)
#
#     func.snake(crop**(0.5),bbox_size/2)
#     # im_=np.asarray(crop,np.float)
#     # im_ = (im_-im_.min())/(2*im_.min()-im_.min())
#     # im_=np.asarray(8*im_,np.uint8)
#     func.snake(crop,bbox_size/2)
##