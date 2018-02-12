from morphsnakes import morphsnakes as smorph
# from chanvese import chanvese3d as cv3d
import numpy as np
import itertools
import h5py
import os
from skimage import io
from skimage.transform import resize
from scipy.sparse import csr_matrix, find
import networkx as nx

def link2pred(linkdata):
    #########################################################
    # convert sub to graph to get upscaled reconstruction
    #########################################################
    numsegments = len(linkdata)
    linkdata_con = np.concatenate(linkdata,axis=0)
    edges = []
    # radius_estimate_around_trace
    for ix in range(numsegments):
        edge1 = linkdata[ix][:-1,-1]
        edge2 = linkdata[ix][1:,-1]
        rad = (linkdata[ix][1:,-2]+linkdata[ix][:-1,-2])/2
        edges.append(np.concatenate((edge1[:,None],edge2[:,None],rad[:,None]),axis=1))

    edges = np.concatenate(edges,axis=0)
    # [keepthese, ia, ic] = unique(edges(:, [1 2]));
    # [subs(:, 1), subs(:, 2), subs(:, 3)] = ind2sub(outsiz([1 2 3]), keepthese);
    # edges_ = reshape(ic, [], 2);
    # weights_ = edges(ia, 3:end);

    # in order to go back to original index: unique_edges[edges_reduced[0,0]]
    unique_edges,unique_indicies,unique_inverse = np.unique(edges[:,:2], return_index=True,return_inverse=True)
    edges_reduced = np.reshape(unique_inverse,(edges.shape[0],2))

    # connectivity graph
    dat = np.ones((edges_reduced.shape[0],1)).flatten()
    e1 = edges_reduced[:,0]
    e2 = edges_reduced[:,1]

    sM = csr_matrix((dat,(e1,e2)), shape=(np.max(edges_reduced)+1,np.max(edges_reduced)+1))
    # build shorthest spanning tree from seed
    seed_location = edges_reduced[0,0]

    nxsM = nx.from_scipy_sparse_matrix(sM)
    # orderlist = nx.dfs_preorder_nodes(nxsM,seed_location)
    # orderlist = np.array(list(orderlist))

    # preds = nx.dfs_predecessors(nxsM,seed_location)
    preds = nx.dfs_predecessors(nxsM)
    preds_np = np.array(list(preds.items()))
    pred_array = np.zeros((preds_np.max()+1,1)).flatten()
    pred_array[preds_np[:,0]]=preds_np[:,1]

    return pred_array

def painth5():
    2

def createBinaryMask(size,point1,point2):
    # creates a box around points
    it=1
def convert2JW(h5file,experiment_folder= './data/JW',number_of_level=3):
    with h5py.File(h5file, "r") as f:
        volume = f["volume"]
        output_dims = volume.shape
        bit_multiplication_array = 2**np.arange(number_of_level)
        target_leaf_size = np.asarray(np.ceil(np.array(output_dims[:3]) / 2 ** number_of_level), np.int)
        padded_size = target_leaf_size*2**number_of_level
        folder_path = []
        range_values = [np.asarray(np.arange(0, padded_size[ii], target_leaf_size[ii]),dtype=np.int).tolist() for ii in range(3)]
        for ix,ref in enumerate(list(itertools.product(*range_values))):
            bb_end = np.asarray(np.min((ref+target_leaf_size,np.array(output_dims[:3])),axis=0),dtype=np.int)
            patch_ = volume[ref[0]:bb_end[0], ref[1]:bb_end[1], ref[2]:bb_end[2], :]

            if np.any(bb_end-np.array(ref) < target_leaf_size):
                # pad
                patch = np.zeros(np.append(target_leaf_size,2),dtype=np.uint16)
                patch[:patch_.shape[0], :patch_.shape[1], :patch_.shape[2],:] = patch_
            else:
                patch = patch_

            # if patch size is smaller than full volume size pad zeros
            folder_inds = np.array(np.unravel_index(ix,([8,8,8])))
            folder_inds = folder_inds + 1
            patch_folder_path = []
            for im in np.arange(number_of_level,0,-1):
                bits = folder_inds>2**(im-1)
                # bit 2 num
                patch_folder_path.append(1+np.sum(bits*bit_multiplication_array))
                folder_inds = folder_inds-2**(im-1)*bits

            # create folder
            outfolder = os.path.join(experiment_folder,'/'.join(str(pp) for pp in patch_folder_path))
            if ~np.any(patch):
                continue

            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            print(outfolder)
            for ichannel in range(2):
                outfile = os.path.join(outfolder,'default.'+str(ichannel)+'.tif')
                io.imsave(outfile, np.swapaxes(patch[:,:,:,ichannel], 2, 0))
def mergeJW(experiment_folder,leaf_size,number_of_level=3):
    # reads an octant, down samples it and save a tif file
    # create all paths from depth-1 to 0
    values = ['{}/'.format(str(ii + 1)) for ii in range(8)]
    for current_number_of_level in np.arange(number_of_level-1,-1,-1):
        for iter_current_folder in list(itertools.product(values, repeat=current_number_of_level)):
            my_lst_str = ''.join(map(str, iter_current_folder))
            current_folder = os.path.join(experiment_folder, my_lst_str)
            if os.path.exists(current_folder):
                print(current_folder)
                iter_octant(current_folder,leaf_size)

def iter_octant(current_folder,leaf_shape):
    for ichannel in range(2):
        im_channel = []
        for ioct in range(8):
            current_path = os.path.join(current_folder,str(ioct+1))
            current_file = current_path+'/default.{}.tif'.format(ichannel)
            if os.path.exists(current_file):
                im_batch = np.swapaxes(io.imread(current_file),2,0)
            else:
                im_batch = np.zeros(leaf_shape)
            im_channel.append(im_batch)

        rt1 = np.concatenate(im_channel[0:2], axis=0)
        rt2 = np.concatenate(im_channel[2:4], axis=0)
        rt3 = np.concatenate(im_channel[4:6], axis=0)
        rt4 = np.concatenate(im_channel[6:8], axis=0)
        rt5 = np.concatenate((rt1,rt2),axis=1)
        rt6 = np.concatenate((rt3,rt4),axis=1)
        merged_Im = np.concatenate((rt5,rt6),axis=2)
        # down sample image by 2
        down_image = resize(merged_Im,leaf_shape,preserve_range=True)
        io.imsave(current_folder+'/default.{}.tif'.format(ichannel),np.asarray(np.swapaxes(down_image,2,0),np.uint16))



# def swapparams():
#     ##################################
#     # run parameter search
#     Nc = 10  # number of points for (0, pi)
#     alphas = np.arange(0.025, 1.5, 0.025)
#     betas = np.arange(0.5, 2.5, 0.5)
#     frangi_cs = np.arange(500, 2500, 500)
#     it=0
#     vals = np.zeros((len(alphas)*len(betas)*len(frangi_cs),len(path_array_indicies)))
#     scales = np.zeros((len(alphas)*len(betas)*len(frangi_cs),len(path_array_indicies)))
#     for alpha, beta, frangi_c in itertools.product(alphas, betas, frangi_cs):
#         print (it,alpha, beta, frangi_c)
#         filtresponse, scaleresponse =frangi.frangi(inputim,
#                                                    sigmas, alpha=alpha, beta=beta, frangi_c=frangi_c, black_vessels=False,
#                                                    window_size = window_size)
#         # sample along recon
#         vals[it,:] = filtresponse.flat[path_array_indicies]
#         scales[it,:] = scaleresponse.flat[path_array_indicies]
#         it +=1
#
#     for alpha, beta, frangi_c in itertools.product(alphas, betas, frangi_cs):
#         print (it,alpha, beta, frangi_c)
#         filtresponse, scaleresponse =frangi.frangi(inputim,
#                                                    sigmas, alpha=alpha, beta=beta, frangi_c=frangi_c, black_vessels=False,
#                                                    window_size = window_size)
#         # sample along recon
#         vals[it,:] = filtresponse.flat[path_array_indicies]
#         scales[it,:] = scaleresponse.flat[path_array_indicies]
#         it +=1
#     # best response is the one that:
#     # * has uniform profile
#     # * has the high filter/scale ratio
#     response = vals/(scales**.5)
#     aa = np.argmax(np.min(response, axis=1) / np.std(response, axis=1))
#     it =0
#     for alpha, beta, frangi_c in itertools.product(alphas, betas, frangi_cs):
#         print (it,alpha, beta, frangi_c)
#         if it==aa:
#             filtresponse_, scaleresponse_ = frangi.frangi(inputim,
#                                                           sigmas, alpha=alpha, beta=beta, frangi_c=frangi_c, black_vessels=False, window_size=window_size)
#             break
#         else:
#             it+=1

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u

def getRadiusIndicies(radius):
    radlist={}
    for rad in radius:
        if rad<1:
            u=np.ones((1,1,1))
        else:
            grid = np.mgrid[list(map(slice, (2*rad+1,2*rad+1,2*rad+1)))].T - rad
            phi = rad - np.sqrt(np.sum((grid.T) ** 2, 0))
            u = np.float_(phi >= 0)
        radlist[rad] = u
    return radlist

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
