import os
import sys
from absl import app
from absl import flags
from absl import logging
import h5py
import numpy as np
from scipy.sparse import csr_matrix, find
from scipy.ndimage.morphology import distance_transform_edt
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
import json
import util
import improc

def print_attrs(name, obj):
    print(name)
    # for key, val in obj.attrs.iteritems():
    #     print('{}:{}'.format(key,val))
def create_test_samples(tilepath,outfile):
    tilepaths=[]
    tilepaths.append('/nrs/mouselight/SAMPLES/2017-09-25-padded/1/8/3/5/8/7')
    tilepaths.append('/nrs/mouselight/SAMPLES/2017-09-25-padded/1/8/3/5/8/8')
    tilepaths.append('/nrs/mouselight/SAMPLES/2017-09-25-padded/3/6/2/2/5/7')
    outfold = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/'

    for iter,tilepath in enumerate(tilepaths):
        print(iter,tilepath)
        im = improc.loadTiles(tilepath).swapaxes(0,2)
        outfile = outfold + 'test-{}.h5'.format(iter)
        with open('/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/axis_tags_raw.json', 'r') as f:
           raw_tag_data = json.load(f)

        with h5py.File(outfile, 'w') as raw_data_h5:
            dset_trace = raw_data_h5.create_dataset("volume", data=im.reshape((1,)+im.shape[:]),
                                                    dtype='uint16',
                                                    compression="gzip",
                                                    compression_opts=9)
            dset_trace.attrs['axistags'] = raw_tag_data.__str__()



def oct_crop(params,oct_path,xyz,patch_half_size = (25,25,25)):
    # divide volume into smaller chunks if needed

    depthBase = params["nlevels"].astype(int)
    tileSize = params["leafshape"].astype(int)

    # list of octpaths for given locations
    octpath, xres = improc.xyz2oct(xyz, params)
    xres_all = np.asarray(xres,np.int)
    tilelist = improc.chunklist(octpath,depthBase) #1..8
    tileids = list(tilelist.keys())
    # box_half_size = (25,25,25)
    invalidTiles = np.any(np.bitwise_or((xres_all-patch_half_size)<0 , (xres_all+patch_half_size)>tileSize),axis=1)
    image_patches=[]
    for iter, idTile in enumerate(tileids):
        print('{} : {} out of {}'.format(idTile, iter, len(tileids)))
        if invalidTiles[iter]:
            continue
        xres_ = xres_all[iter]
        tilename = '/'.join(a for a in idTile)
        tilepath = oct_path + '/' + tilename
        im = improc.loadTiles(tilepath)
        # crop patch around res
        image_patches.append(crop_patch(im, xres_, patch_half_size))
        # sitk.Show(sitk.GetImageFromArray(np.swapaxes(im[:,:,:,0], 2, 0)))
    return image_patches

def crop_negative_patches_from_oct(oct_path='/nrs/mouselight/SAMPLES/2017-09-25-padded',
                                   coord_list='/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/negative_sample_coordinates.txt',
                                   patch_half_size=(25, 25, 25)):
    params = util.readParameterFile(parameterfile=oct_path+"/calculated_parameters.jl")
    scale = 1/1000
    with open(coord_list, 'r') as f:
        um = f.readlines()
        um = np.array([eval(x.strip()) for x in um])
    # um, edges, R, offset, scale, header = util.readSWC(swcfile=coord_list,scale=1)

    # to fix the bug in Janelia Workstation
    um = um + params['vixsize']/2
    nm =um/scale
    xyz = util.um2pix(nm,params['A']).T
##
    neg_patches = oct_crop(params, oct_path, xyz, patch_half_size=patch_half_size)
    return np.stack(neg_patches,axis=0)




def crop_patch(img,pos_sub,patch_half_size):
    if img.ndim>3:
        crop = img[pos_sub[0] - patch_half_size[0]:pos_sub[0] + patch_half_size[0]+1,
               pos_sub[1] - patch_half_size[1]:pos_sub[1] + patch_half_size[1]+1,
               pos_sub[2] - patch_half_size[2]:pos_sub[2] + patch_half_size[2]+1, :]
    else:
        crop = img[pos_sub[0] - patch_half_size[0]:pos_sub[0] + patch_half_size[0] + 1,
               pos_sub[1] - patch_half_size[1]:pos_sub[1] + patch_half_size[1] + 1,
               pos_sub[2] - patch_half_size[2]:pos_sub[2] + patch_half_size[2] + 1]
    return crop


def main(argv):
    # training samples:
    #   key nodes (labels):
    #       i) junctions (+1)
    #       ii) tips (+2)
    #       iii) regular locations (+3)
    #       iv) inverse-masked background (-1) : for each + sample patch, sample - samples from mask inverse (make sure patch is in bounding box)
    #       v) auto-florescence (-2)
    #       vi) ventricles (-3)

    flip_axis = True # to compansate a bug in 2017-09-25 JW workspaces
    user_input = True
    patch_half_size=(25,25,25)
    p = 0.2 # negative examples sampring rate. p for segment-/+epsilon bound, (1-p) from random background region
    if user_input:
        oct_path = '/nrs/mouselight/SAMPLES/2017-09-25-padded'
        coord_list = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/negative_sample_coordinates.txt'
        neg_samples_raw = crop_negative_patches_from_oct(oct_path, coord_list,patch_half_size)
        neg_samples_label = np.zeros(neg_samples_raw.shape[:-1],np.uint8)+2
        # mute
        mask = np.random.choice([0, 1], size=neg_samples_label.shape[1::], p=[.99, .01])
        mask[patch_half_size[0],patch_half_size[1],patch_half_size[2]] = 1
        for ii in range(neg_samples_label.shape[0]):
            neg_samples_label[ii, :, :, :] = neg_samples_label[ii, :, :, :] * mask

    with open('/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/axis_tags_label.json', 'r') as f:
       label_tag_data = json.load(f)
    with open('/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/axis_tags_raw.json', 'r') as f:
       raw_tag_data = json.load(f)


    # datasets:
    input_h5 = h5py.File('/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/2017-09-25_G-007_consensus-annotation.h5','r')
    input_h5.visititems(print_attrs)
    raw_data = input_h5['volumes']['raw']
    segmentation_data = input_h5['volumes']['segmentation']
    trace_data = input_h5['volumes']['trace']

    # read recon data
    recon = input_h5['reconstructions']['sparse']
    edges = recon[:,(0,6)]
    edges = np.delete(edges,np.where(np.any(edges==-1,axis=1)),0)
    edges = edges-1
    subs = recon[:,2:5]
    # connectivity graph
    dat = np.ones((edges.shape[0],1)).flatten()
    e1 = edges[:,0]
    e2 = edges[:,1]
    sM = csr_matrix((dat,(e1,e2)), shape=(np.max(edges)+1,np.max(edges)+1))

    plt.plot(subs[:,0],subs[:,1])

    # junctions
    junction_nodes = np.where(np.sum(sM,axis=0)>1)[1]
    tip_nodes = np.where(np.sum(sM,axis=0)==0)[1]
    regular_nodes = np.where(np.sum(sM,axis=0)==1)[1]
    bins = np.cumsum([0, len(junction_nodes), len(tip_nodes), len(regular_nodes)])
    bin_array = np.zeros((1,bins[-1]))
    for iter,it in enumerate(bins[:-1]):
        print(iter,it)
        bin_array[0,bins[iter]:bins[iter+1]] = iter+1

    positive_labels = np.full((1,np.asarray(np.max(edges),np.int)+1),np.nan)
    positive_labels[0,junction_nodes] = 1
    positive_labels[0,tip_nodes] = 2
    positive_labels[0,regular_nodes] = 3
    positive_subs = subs[np.concatenate((junction_nodes,tip_nodes,regular_nodes),axis=0),:]

    rawstack = []
    sparselabelstack = []
    denselabelstack = []
    stats=[]
    node_type = []
    for iter,pos_sub in enumerate(positive_subs[:,:]):
        print(iter,'out of', positive_subs.shape[0])
        #crop a patch around the sub
        pos_sub = np.asarray(pos_sub,np.int)
        if np.any(pos_sub-patch_half_size<0) or np.any(pos_sub+patch_half_size>raw_data.shape[:3]): # out of bound
            continue

        node_type.append(bin_array[0,iter])

        if flip_axis:
            raw_patch = np.flip(crop_patch(raw_data,pos_sub,patch_half_size),axis=3)
        else:
            raw_patch = crop_patch(raw_data,pos_sub,patch_half_size)

        trace_patch = crop_patch(trace_data,pos_sub,patch_half_size)

        segment_patch = crop_patch(segmentation_data,pos_sub,patch_half_size)

        stack = np.stack((raw_patch[:,:,:,0],trace_patch,segment_patch),axis=3)

        # # sitk.Show(sitk.GetImageFromArray(np.swapaxes(label_patch, 2, 0)))
        # i1 = sitk.GetImageFromArray(np.swapaxes(raw_patch[:,:,:,0],2,0))
        # i2 = sitk.GetImageFromArray(np.swapaxes(trace_patch[:,:,:],2,0))
        # i3 = sitk.GetImageFromArray(np.swapaxes(segment_patch[:,:,:],2,0))
        # # i123 = sitk.LabelOverlay(i1, i2)
        # sitk.Show(i1)
        # sitk.Show(i2)

        # 3D patch for ilastik tzyxc format
        rawstack.append(raw_patch.swapaxes(0,2))

        # sparse annotation
        label_patch = 1*trace_patch
        num_pos_label = np.sum(label_patch)

        # sample from background
        dist = distance_transform_edt(1-segment_patch)
        neg_label1 = np.bitwise_and(dist > 4, dist < 6)
        neg_label2 = dist > 6

        # with rate p, sample from neg_1
        neg_label1_locs = np.array(np.where(neg_label1))
        neg_label2_locs = np.array(np.where(neg_label2))
        neg_label_locs = np.concatenate((neg_label1_locs,neg_label2_locs),axis=1)

        rand_neg_samp = random.sample(range(neg_label1_locs.shape[1]), np.int(p*num_pos_label)) + \
                        random.sample(range(neg_label2_locs.shape[1]), np.int((1-p)*num_pos_label))

        neg_sub = neg_label_locs[:, rand_neg_samp]
        indicies = np.ravel_multi_index(neg_sub, label_patch.shape)

        # paint negative labels
        label_patch.flat[indicies] = 2
        sparselabelstack.append(label_patch)

        tr=raw_patch[:,:,:,0]
        rt=[np.min(tr[label_patch == 1]),np.median(tr[label_patch == 1])]
        tr=raw_patch[:,:,:,1]
        rt+=[np.min(tr[label_patch == 1]),np.median(tr[label_patch == 1])]
        stats.append(rt)

        #dense annotation
        denselabelstack.append(1*segment_patch)



    stats_ = np.array(stats)
    node_type_ = np.array(node_type)

    rawstack = np.stack(rawstack,axis=0)
    sparselabelstack = np.stack(sparselabelstack,axis=0)
    denselabelstack = np.stack(denselabelstack,axis=0)

    reg_node_indicies = np.where(node_type_==3)[0]
    reg_stats_ = stats_[reg_node_indicies,1] # 3: regular node
    aa = np.argsort(reg_stats_)

    selected_regnode_indicies = reg_node_indicies[aa[np.asarray(np.linspace(0, len(aa)-1, len(junction_nodes) + len(tip_nodes)), np.int)]]
    selected_indicies = np.concatenate((np.arange(len(junction_nodes) + len(tip_nodes)),selected_regnode_indicies),axis=0)

    rawstack = rawstack[selected_indicies,:,:,:,:]
    sparselabelstack = sparselabelstack[selected_indicies,:,:,:]
    denselabelstack = denselabelstack[selected_indicies,:,:,:]

    i3 = sitk.GetImageFromArray(np.swapaxes(rawstack[241, :, :, :, 0], 0, 0))
    sitk.Show(i3)

    # dump to h5
    out_raw_h5 = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/2017-09-25_G-007_consensus-training_raw.h5'
    out_raw_h5_u8 = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/2017-09-25_G-007_consensus-training_raw_u8.h5'
    out_sparse_label_h5 = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/2017-09-25_G-007_consensus-training_sparse_label.h5'
    out_dense_label_h5 = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/2017-09-25_G-007_consensus-training_dense_label.h5'

    # with h5py.File(out_raw_h5_u8,'w') as raw_data_h5:
    #     rawstack_ = np.array(rawstack, copy=True,dtype=np.float)
    #     rawstack_ = 255*(rawstack_ - rawstack_.min())/(rawstack_.max() - rawstack_.min())
    #     rawstack_ = np.asarray(rawstack_,dtype=np.uint8)
    #     dset_trace = raw_data_h5.create_dataset("volume", data=rawstack_,
    #                                    dtype='uint8',
    #                                    compression="gzip",
    #                                    compression_opts=9)
    #     dset_trace.attrs['axistags'] = raw_tag_data.__str__()

    # concatenate neg label
    if user_input:
        rawstack = np.concatenate((rawstack,neg_samples_raw),axis=0)
        sparselabelstack = np.concatenate((sparselabelstack,neg_samples_label),axis=0)
        denselabelstack = np.concatenate((denselabelstack,neg_samples_label),axis=0)

    with h5py.File(out_raw_h5, 'w') as raw_data_h5:
        dset_trace = raw_data_h5.create_dataset("volume", data=rawstack,
                                   dtype='uint16',
                                   compression="gzip",
                                   compression_opts=9)
        dset_trace.attrs['axistags'] = raw_tag_data.__str__()

    with h5py.File(out_sparse_label_h5, 'w') as label_data_h5:
        dset_trace = label_data_h5.create_dataset("volume", data=sparselabelstack,
                                                dtype='uint8',
                                                compression="gzip",
                                                compression_opts=9)
        dset_trace.attrs['axistags'] = label_tag_data.__str__()

    with h5py.File(out_dense_label_h5, 'w') as label_data_h5:
        dset_trace = label_data_h5.create_dataset("volume", data=denselabelstack,
                                                dtype='uint8',
                                                compression="gzip",
                                                compression_opts=9)
        dset_trace.attrs['axistags'] = label_tag_data.__str__()


if __name__ == "__main__":
   main(sys.argv[1:])


# main(sys.argv[1:])