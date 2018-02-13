import os
import sys, getopt
import numpy as np
import h5py
from frangi3d import frangi
import importlib
import functions as func
from skimage import graph as skig
from scipy.ndimage.filters import median_filter,minimum_filter
# sys.path.insert(0, '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/lineScanFix')
import segment as seg

# need it for debug/viz
import SimpleITK as sitk
import matplotlib.pyplot as plt

def main(argv):
    generate_output = True
    input_h5_file = '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/navigator/data/2017-11-17_G-017_Seg-1/2017-11-17_G-017_Seg-1-cropped.h5'
    output_folder = '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/navigator/data/2017-11-17_G-017_Seg-1/'
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["input_h5_file=","output_folder="])
    except getopt.GetoptError:
        print('segmentAxon.py -i <input_h5_file> -o <output_folder>')
        sys.exit(2)
    for opt, arg in opts:
        print('opt:', opt,'arg:', arg)
        if opt == '-h':
            print('segmentAxon.py -i <input_h5_file> -o <output_folder>')
            sys.exit()
        elif opt in ("-i", "--input_h5_file"):
            input_h5_file = arg
        elif opt in ("-o", "--output_folder"):
            output_folder = arg

    print('SWCFILE   :', input_h5_file)
    print('OUTPUT    :', output_folder)

    # swc_name = '2017-11-17_G-017_Seg-1'
    # data_folder = os.path.join('/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/navigator/data/',swc_name)

    inputfolder,h5_name_w_ext = os.path.split(input_h5_file)
    file_name,_ = h5_name_w_ext.split(os.extsep)

    segmentation_output_h5_file = os.path.join(output_folder,file_name+'_segmented.h5')
    swc_output_file = os.path.join(output_folder,file_name+'_segmented.swc')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # AC_output_tif_file =  os.path.join(data_folder,swc_name+'_AC_cropped_segmented.tif')
    # Frangi_output_tif_file =  os.path.join(data_folder,swc_name+'_Frangi_cropped_segmented.tif')


    # TODO export scale/filt if needed
    # dset_filter_Frangi_magnitude = f_out.create_dataset("/filter/Frangi/magnitude", volume.shape[:3], dtype='f', chunks=f["volume"].chunks[:3], compression="gzip", compression_opts=9)
    # dset_filter_Frangi_scale = f_out.create_dataset("/filter/Frangi/scale", volume.shape[:3], dtype='f', chunks=f["volume"].chunks[:3], compression="gzip", compression_opts=9)
    # for each branch, crop a box, run segmentation based on:
    # 1) frangi vesselness filter
    # 2) active countours
    # 3) stat thresholding: TODO: diffusion is buggy, might be better to switch to a regularized version

    with h5py.File(input_h5_file, "r") as f:
        recon = f["reconstruction"]
        volume = f["volume"]
        output_dims = volume.shape
        if generate_output:
            f_out = h5py.File(segmentation_output_h5_file, "w")
            dset_segmentation_AC = f_out.create_dataset("/segmentation/AC", volume.shape[:3], dtype='uint8',
                                                        chunks=f["volume"].chunks[:3], compression="gzip",
                                                        compression_opts=9)
            dset_segmentation_Frangi = f_out.create_dataset("/segmentation/Frangi", volume.shape[:3], dtype='uint8',
                                                            chunks=f["volume"].chunks[:3], compression="gzip",
                                                            compression_opts=9)

        lookup_data={} # keeps track of indicies, subs and radius
        sigmas = np.array([0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
        linkdata = []
        window_size = 3
        radius_list = func.getRadiusIndicies(radius=sigmas)

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
            # filter out scale response for local radius variations:
            cost_array = minimum_filter(scaleresponse,2)/(0.001+filtresponse)
            cost_array[cost_array>100]=100

            # shortest path based on cost_array
            path,cost = skig.route_through_array(cost_array, start=start_, end=end_, fully_connected=True)
            path_array = np.asarray(path)

            # sample along path
            path_array_indicies = np.ravel_multi_index(path_array.T,cost_array.shape)

            # frangi radius estimate around tracing
            radius_estimate_around_trace = scaleresponse.flat[path_array_indicies]
            # filter radius to smooth
            radius_estimate_around_trace = median_filter(radius_estimate_around_trace,3)
            # print(np.max(radius_estimate_around_trace))

            # fine tuned swc file
            xyz_trace_locations = path_array + bbox_min

            # plt.figure()
            # plt.imshow(np.max(filtresponse**.05,axis=2).T)
            # plt.plot(path_array[:,0],path_array[:,1])

            # index ids for each location
            inds = np.ravel_multi_index(xyz_trace_locations.T,output_dims[:3])

            # radius as 4th column
            branch_data = np.concatenate((xyz_trace_locations,radius_estimate_around_trace[:,None],inds[:,None]),axis=1)

            for ii,ind in enumerate(inds):
                lookup_data[ind] = branch_data[ii,:4]

            # store recon info
            linkdata.append(branch_data)

            if generate_output: # paint functions
                # paint hdf5: for each trace location, generate a ball with the given radius and paint into segmentation output
                for xyzr in branch_data:
                    xyz=xyzr[:3]
                    r=xyzr[3]
                    # search the nearest key
                    mask = radius_list[sigmas[np.argmin((sigmas-r)**2)]]
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

                # patch wise write is buggy, so location wise painting
                xyz_signal = bbox_min[:,None] + np.where(segment.mask_ActiveContour)
                for xyz in xyz_signal.transpose():
                    dset_segmentation_AC[tuple(xyz)] = 1

        swc_data = np.array(func.link2pred(linkdata,lookup_data))
        func.array2swc(swcfile=swc_output_file, swcdata=swc_data)
        if generate_output:
            dset_swc_Frangi = f_out.create_dataset("/trace/trace", data=swc_data, dtype='f')
            f_out.close()

if __name__ == "__main__":
   main(sys.argv[1:])