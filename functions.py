from morphsnakes import morphsnakes as smorph
import numpy as np

from scipy.sparse import csr_matrix, find
import networkx as nx

def array2swc(swcfile,swcdata):
    with open(swcfile,'w') as fswc:
        for iter,txt in enumerate(swcdata[:,:]):
            fswc.write('{:.0f} {:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.0f}\n'.format(txt[0],txt[1],txt[2],txt[3],txt[4]-1,txt[5],txt[6]))

def link2pred(linkdata,lookup_data):
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
    seed_index = edges_reduced[0,0]

    nxsM = nx.from_scipy_sparse_matrix(sM)

    preds = nx.dfs_predecessors(nxsM,seed_index)
    orderlist = nx.dfs_preorder_nodes(nxsM, seed_index)
    orderlist = np.array(list(orderlist))
    seed_vals = lookup_data[unique_edges[seed_index]]

    swc_data=[]
    swc_list={}
    # iterate over orderlist (set first column based on this)
    for ix, idx_trace in enumerate(orderlist):
        swc_list[idx_trace] = ix + 1
        if ix==0:
            target = -1
        else:
            target = swc_list[preds[idx_trace]]

        loc_xyzr = lookup_data[unique_edges[idx_trace]]
        swc_data.append([ix+1,1,loc_xyzr[0],loc_xyzr[1],loc_xyzr[2],loc_xyzr[3],target])

    return swc_data

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
