# import util
import numpy as np
import matplotlib.pyplot as plt
# import improc
# import os
# from scipy.spatial import distance as dist
import h5py
# importlib.reload(frangi)
from frangi3d import frangi
import importlib
import functions as func
from skimage import exposure
from skimage import graph as skig
from skimage import io
import sys
sys.path.insert(0, '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/lineScanFix')
from scipy.sparse import csr_matrix, find
import networkx as nx
import segment as seg
import os

from sklearn import feature_extraction as feat
from scipy.sparse import csgraph as csg
from sklearn.cluster import spectral_clustering
import SimpleITK as sitk
import itertools
# import linefix as lnf
from scipy.sparse import coo_matrix,csc_matrix

def test(volume,recon):
# format(recon[iter, 0].__int__(), 1, txt[0], txt[1], txt[2], 1, recon[iter, 1].__int__()))

    with open('./data/neuron_test-1.swc','w') as fswc:
        for iter,txt in enumerate(recon[:,:]):
            fswc.write('{:.0f} {:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.0f}\n'.format(txt[0],txt[1],txt[2]-1,txt[3]-1,txt[4]-1,txt[5],txt[6]))

    io.imsave('./data/neuron_test.tif',np.swapaxes(volume[:,:,:,0],2,0))



if __name__ == '__main__':
    # this is the main
    generate_output = False
    data_folder = '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/navigator/data'
    swc_name = '2017-11-17_G-017_Seg-1'
    swc_file = os.path.join(data_folder,swc_name+'.swc')
    cropped_swc_file = os.path.join(data_folder,swc_name+'_cropped.swc')
    cropped_h5_file =  os.path.join(data_folder,swc_name+'_cropped.h5')
    cropped_h5_file_output =  os.path.join(data_folder,swc_name+'_cropped_segmented.h5')
    AC_cropped_tif_file_output =  os.path.join(data_folder,swc_name+'_AC_cropped_segmented.tif')
    Frangi_cropped_tif_file_output =  os.path.join(data_folder,swc_name+'_Frangi_cropped_segmented.tif')

    f = h5py.File(cropped_h5_file, "r")
    recon = f["reconstruction"]
    volume = f["volume"]
    output_dims = volume.shape

    if generate_output:
        f_out = h5py.File(cropped_h5_file_output, "w")
        dset_segmentation_AC = f_out.create_dataset("/segmentation/AC", volume.shape[:3], dtype='uint8', chunks=f["volume"].chunks[:3], compression="gzip", compression_opts=9)
        dset_segmentation_Frangi = f_out.create_dataset("/segmentation/Frangi", volume.shape[:3], dtype='uint8', chunks=f["volume"].chunks[:3], compression="gzip", compression_opts=9)
        dset_swc_Frangi = f_out.create_dataset("/trace/trace", (), dtype='f')

    # TODO export scale/filt if needed
    # dset_filter_Frangi_magnitude = f_out.create_dataset("/filter/Frangi/magnitude", volume.shape[:3], dtype='f', chunks=f["volume"].chunks[:3], compression="gzip", compression_opts=9)
    # dset_filter_Frangi_scale = f_out.create_dataset("/filter/Frangi/scale", volume.shape[:3], dtype='f', chunks=f["volume"].chunks[:3], compression="gzip", compression_opts=9)
    # for each branch, crop a box, run segmentation based on:
    # 1) frangi vesselness filter
    # 2) active countours
    # 3) stat thresholding: TODO: diffusion is buggy, might be better to switch to a regularized version

    linkdata = []
    sigmas = np.array([0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    window_size = 3

    radius_list = func.getRadiusIndicies(radius=sigmas)
    lookup_data={}
    for ix, txt in enumerate(recon[:, :]):
        print('{} out of {}'.format(ix,recon.shape[0]))
        start_node = ix #  recon[ix,0]
        end_node = recon[ix,6]-1 # 0 based indexing
        if end_node < 0:
            continue

        start = np.asarray(recon[start_node,2:5],np.int)
        end = np.asarray(recon[end_node,2:5],np.int)

        # vesselness convolution sigmas
        # padding needs to be at least window_size/2*sigma
        padding = np.ceil(np.max(window_size*sigmas/2)).__int__()

        bbox_min_wo = np.min((start, end), axis=0)
        bbox_max_wo = np.max((start, end), axis=0)
        bbox_min = np.min((start,end),axis=0)-padding
        bbox_max = np.max((start,end),axis=0)+padding+1 # add one to make python inclusive

        bbox_size = bbox_max-bbox_min
        start_ = start - bbox_min
        end_ = end-bbox_min
        roi = ((padding),())

        # crop
        crop = volume[bbox_min[0]:bbox_max[0],bbox_min[1]:bbox_max[1],bbox_min[2]:bbox_max[2]]
        crop[crop==0] = np.min(crop[crop>0]) # overwrites any missing voxel with patch minima

        ## fix line shifts
        # st = -9;en = 10;shift, shift_float = lnf.findShift(inputim[:,:inputim.shape[1]//2*2,:], st, en, False)
        ##
        inputim = np.log(np.asarray(crop[:,:,:,0], np.float)) # add 1 to prevent /0 cases for log scaling
        inputim =(inputim-inputim.min())/(inputim.max()-inputim.min())

        ## FRANGI
        filtresponse, scaleresponse =frangi.frangi(inputim, sigmas,window_size = window_size,
                                                   alpha=0.01, beta=1.5, frangi_c=2*np.std(inputim), black_vessels=False)

        # sitk.Show(sitk.GetImageFromArray(np.swapaxes(inputim,2,0)))
        # sitk.Show(sitk.GetImageFromArray(np.swapaxes(filtresponse/np.max(filtresponse),2,0)))
        # sitk.Show(sitk.GetImageFromArray(np.swapaxes(scaleresponse,2,0)))

        # cost: high intensity low scale
        cost_array = scaleresponse/(0.001+filtresponse)
        cost_array[cost_array>100]=100

        # shortest path based on cost_array
        path,cost = skig.route_through_array(cost_array, start=start_, end=end_, fully_connected=True)
        path_array = np.asarray(path)

        # sample along path
        path_array_indicies = np.ravel_multi_index(path_array.T,cost_array.shape)

        # frangi radius estimate around tracing
        radius_estimate_around_trace = scaleresponse.flat[path_array_indicies]
        # print(np.max(radius_estimate_around_trace))

        # fine tuned swc file
        xyz_trace_locations = path_array + bbox_min

        # index ids for each location
        inds = np.ravel_multi_index(xyz_trace_locations.T,output_dims[:3])

        # radius as 4th column
        branch_data = np.concatenate((xyz_trace_locations,radius_estimate_around_trace[:,None],inds[:,None]),axis=1)

        for ii,ind in enumerate(inds):
            lookup_data[ind] = branch_data[ii,:4]

        # store recon info
        linkdata.append(branch_data)
        if 0: # paint functions
            # paint hdf5: for each trace location, generate a ball with the given radius and paint into segmentation output
            for xyzr in branch_data:
                xyz=xyzr[:3]
                r=xyzr[3]
                mask = radius_list[r]
                paintlocs = np.where(mask)-np.floor(r) + xyz[:,None]
                for locs in paintlocs.transpose():
                    dset_segmentation_Frangi[tuple(locs)] = 1

            ## segmentation based on active contours
            if 1:
                segment = seg.volumeSeg(filtresponse,path_array) # working
            else: # use cost function to initialize segmentation
                segment = seg.volumeSeg(inputim,path_array,cost_array=np.max(cost_array)-cost_array) # revert cost for positive active contour

            segment.runSeg()
            # sitk.Show(sitk.GetImageFromArray(np.swapaxes(segment.mask_ActiveContour,2,0)))
            # sitk.Show(sitk.GetImageFromArray(np.swapaxes(inputim,2,0)))
            # sitk.Show(sitk.GetImageFromArray(np.swapaxes(filtresponse/np.max(filtresponse),2,0)))

            # paint segmentation result
            # patch wise write is buggy:
            # dset_segmentation_AC[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]] = segment.mask_ActiveContour # results in boundary artifacts
            # dset_segmentation_AC[bbox_min_wo[0]:bbox_min_wo[0], bbox_min_wo[1]:bbox_min_wo[1], bbox_min_wo[2]:bbox_min_wo[2]] = segment.mask_ActiveContour[padding:-padding,padding:-padding,padding:-padding] # results in missing data
            # location wise painting
            xyz_signal = bbox_min[:,None] + np.where(segment.mask_ActiveContour)
            for xyz in xyz_signal.transpose():
                dset_segmentation_AC[tuple(xyz)] = 1


    f.close()
    if generate_output:
        f_out.close()

    pred_array = func.link2swc(linkdata)


    # with open(os.path.join(data_folder,swc_name+'-upsampled.swc'),'w') as fswc:
    #     for ipx,ip in enumerate(pred_array):
    #         curr_node = lookup_data[unique_edges[ipx]]
    #
    #         if ipx ==0:
    #             str = '{:.0f} {:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.0f}\n'.format(ipx+1,1,curr_node[0],curr_node[1],curr_node[2],curr_node[3],-1)
    #         else:
    #             str = '{:.0f} {:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.0f}\n'.format(ipx+1,1,curr_node[0],curr_node[1],curr_node[2],curr_node[3],ip+1)




    # test_im = volume[:,:,:100,0]
    # nm = np.max(test_im,axis=2)
    # plt.figure()
    # plt.imshow(nm.T)
    # T = nx.dfs_tree(nxsM, seed_location)
    # for edT in sorted(T.edges(data=True)):
    #     from_node = lookup_data[unique_edges[edT[0]]]
    #     to_node = lookup_data[unique_edges[edT[1]]]
    #     # draw en edge
    #     plt.plot([from_node[0],to_node[0]],[from_node[1],to_node[1]])

    # #convert to tif
    # with h5py.File(cropped_h5_file_output, "r") as f:
    #     dset_segmentation_AC = f['/segmentation/AC']
    #     io.imsave(AC_cropped_tif_file_output, np.swapaxes(dset_segmentation_AC,2,0))
    # with h5py.File(cropped_h5_file_output, "r") as f:
    #     dset_segmentation_AC = f['/segmentation/Frangi']
    #     io.imsave(Frangi_cropped_tif_file_output, np.swapaxes(dset_segmentation_AC,2,0))

    # # convert 3D stack to octree format
    # experiment_folder = os.path.join(data_folder, 'JW', swc_name)
    # number_of_level = 3
    # target_leaf_size = np.asarray(np.ceil(np.array(output_dims[:3]) / 2 ** number_of_level), np.int)
    # importlib.reload(func)
    # func.convert2JW(cropped_h5_file,experiment_folder=experiment_folder,number_of_level=number_of_level)
    # func.mergeJW(experiment_folder,target_leaf_size) # reads folders and generates down sampled version


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
