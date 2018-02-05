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
from skimage import exposure
from sklearn import feature_extraction as feat
from skimage import graph as skig
from scipy.sparse import csgraph as csg
from sklearn.cluster import spectral_clustering
import SimpleITK as sitk
from skimage import io

import segment as seg
import os
def test(volume,recon):
# format(recon[iter, 0].__int__(), 1, txt[0], txt[1], txt[2], 1, recon[iter, 1].__int__()))

    with open('./data/neuron_test-1.swc','w') as fswc:
        for iter,txt in enumerate(recon[:,:]):
            fswc.write('{:.0f} {:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.0f}\n'.format(txt[0],txt[1],txt[2]-1,txt[3]-1,txt[4]-1,txt[5],txt[6]))

    io.imsave('./data/neuron_test.tif',np.swapaxes(volume[:,:,:,0],2,0))


if __name__ == '__main__':
    # this is the main
    datafolder = '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/navigator/data'
    swcname = '2017-11-17_G-017_Seg-1'
    swc_file = os.path.join(datafolder,swcname+'.swc')
    cropped_swc_file = os.path.join(datafolder,swcname+'_cropped.swc')
    cropped_h5_file =  os.path.join(datafolder,swcname+'_cropped.h5')
    cropped_tif_file =  os.path.join(datafolder,swcname+'_cropped.tif')

    # with h5py.File(h5file, "r") as f:
    f = h5py.File(cropped_h5_file, "r")
    recon = f["reconstruction"]
    volume = f["volume"]
# for each branch, crop a box, run segmentation based on:
    # 1) frangi vesselness filter
    # 2) stat thresholding
    # 3) active countours
    for ix, txt in enumerate(recon[:, :]):
        ix=0
        start = np.asarray(recon[ix,2:5],np.int)
        end = np.asarray(recon[ix+1,2:5],np.int)

        # vesselness convolution sigmas
        sigmas = np.array([0.5,1.0,1.5,2,2.5,3,5,7,10])
        window_size = 3
        # padding needs to be at least window_size/2*sigma
        padding = np.max(window_size*sigmas/2).__int__()

        bbox_min = np.min((start,end),axis=0)-padding
        bbox_max = np.max((start,end),axis=0)+padding+1 # add one to make python inclusive
        bbox_size = bbox_max-bbox_min
        start_ = start - bbox_min
        end_ = end-bbox_min
        roi = ((padding),())

        # crop
        crop = volume[bbox_min[0]:bbox_max[0],bbox_min[1]:bbox_max[1],bbox_min[2]:bbox_max[2]]
        # crop = volume[bbox_min[0]:bbox_max[0],bbox_min[1]:bbox_max[1],375,0]
        # f, (ax1) = plt.subplots(1, 1)
        # ax1.imshow(crop.T, cmap='gray')
        # swap x-z for visualization
        if False:
            crop = crop.swapaxes(0,2)
            bbox_size = bbox_size[[2,1,0]]
            start_ = start_[[2,1,0]]
            end_ = end_[[2,1,0]]

        inputim = np.asarray(crop[:,:,:,0], np.float)
        ##

        importlib.reload(frangi)
        filtresponse, scaleresponse =frangi.frangi(inputim,
                                                   sigmas, alpha=0.05, beta=1.5, frangi_c=2000, black_vessels=False,
                                                   window_size = window_size)

        # cost: high intensity low scale
        cost_array = scaleresponse/(0.001+filtresponse)
        cost_array[cost_array>100]=100

        cost_array = sitk.GetArrayFromImage(sitk.BinaryThreshold(sitk.GetImageFromArray(cost_array),0,100,outsideValue=0))

        path,cost = skig.route_through_array(cost_array, start=start_, end=end_, fully_connected=True)
        path_array = np.asarray(path)
        # sample along path
        path_array_indicies = np.ravel_multi_index(path_array.T,cost_array.shape)

        importlib.reload(seg)
        segment = seg.volumeSeg(filtresponse,path_array,cost_array)
        segment.runSeg()

        sitk.Show(sitk.GetImageFromArray(np.swapaxes(inputim,2,0)))
        sitk.Show(sitk.GetImageFromArray(np.swapaxes(scaleresponse,2,0)))
        sitk.Show(sitk.GetImageFromArray(np.swapaxes(filtresponse,2,0)))
        sitk.Show(sitk.GetImageFromArray(np.swapaxes(segment.mask_Threshold,2,0)))
        sitk.Show(sitk.GetImageFromArray(np.swapaxes(segment.mask_ActiveContour,2,0)))


    sitk.Show(sitk.GetImageFromArray(np.swapaxes(cost_array,2,0)))

    sitk.Show(fastMarching_image)

    # f, (ax1) = plt.subplots(1, 1)
    # ax1.imshow(crop.T, cmap='gray')




    if 0:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        ax1.imshow(np.max(inputim, axis=2).T, cmap='gray')
        ax1.set_title('input_image')

        ax2.imshow(np.max(filtresponse, axis=2).T, cmap='gray')
        ax2.set_title('filter_image')

        gamma_corrected = exposure.adjust_gamma(filtresponse, .5)
        ax3.imshow(np.max(gamma_corrected, axis=2).T, cmap='gray')
        ax3.set_title('gamma_corrected_image')

        ax4.imshow(np.max(scaleresponse * (filtresponse > .1), axis=2).T, cmap='gray')
        ax4.set_title('thresholded_image')

        plt.close('all')
        f, (ax1,ax2) = plt.subplots(1, 2)
        ax1.imshow(np.max(filtresponse, axis=2).T, cmap='gray')
        ax1.plot(path_array[:,0],path_array[:,1])
        ax1.set_title('filter_image')
        ax2.imshow(-np.min(cost_array, axis=2).T, cmap='gray')
        ax2.plot(path_array[:,0],path_array[:,1])
        ax2.set_title('cost_image')


    # # search between nodes by traversing a weighted graph
    # # convert image to weighted graph
    # filter_graph = feat.img_to_graph(filtresponse)
    # scale_graph = feat.img_to_graph(scaleresponse)
    # filter_graph.data = np.exp(-filter_graph.data / filter_graph.data.std())
    # n_clusters = 3
    # labels = spectral_clustering(filter_graph, n_clusters=n_clusters, eigen_solver='arpack')
    # labimage = labels.reshape(filtresponse.shape)
    # f, (ax1) = plt.subplots(1, 1)
    # ax1.imshow(np.max(labimage, axis=2).T, cmap='gray')

    # ##
#     mask = inputim*0
#     start_point = start_
#     start_point = np.array([7,1,3])
#     import functions as func
#
#     importlib.reload(func)
#     seg = func.region_grow(inputim, mask, start_point, epsilon=1000, HU_mid=inputim[start_point[0],start_point[1],start_point[2]], HU_range=1000, fill_with=255)
#
#     plt.figure()
#     for ii in range(5):
#         ax=plt.subplot(1,5,ii+1)
#         ax.imshow(inputim[:,:,ii].T, cmap='gray', interpolation='nearest')
#         ax.imshow(mask[:, :, ii].T, cmap='jet', interpolation='nearest', alpha=.3)
#         ax.set_title(ii)
#         # ax.tick_params(labeltop=True, labelright=True)
# ##

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

    # inputimage = sitk.GetImageFromArray(filtresponse)
    # seeds = (path_array[:, 0], path_array[:, 1], path_array[:, 2])
    #
    # seg = sitk.Image(inputimage.GetSize(), sitk.sitkUInt8)
    # seg.CopyInformation(inputimage)
    # for idx,seed in enumerate(seeds):
    #     seg[seed.tolist()] = 1
    #
    # # Binary dilate enlarges the seed mask by 3 pixels in all directions.
    # seg = sitk.BinaryDilate(seg, radius)
    # init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)
    #
    # sitk.Show(init_ls)
