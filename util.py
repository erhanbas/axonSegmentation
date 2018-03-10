import numpy as np
import itertools
import os
from skimage import io
import h5py
from skimage.transform import resize
import warnings


from collections import defaultdict
import improc

def readTransfrom(transformfile = "/nrs/mouselight/SAMPLES/2017-06-10/transform.txt"):
    # reads transform.txt file and parse it into a transform
    A = np.zeros((3,4))
    with open(transformfile, 'r') as f:
        while True:
            text = f.readline()
            if not text: break
            parts = text.split(':')
            num = np.float(parts[1].strip('\n'))
            if parts[0] == 'ox':
                A[0,3] = num
            elif parts[0] == 'oy':
                A[1, 3] = num
            elif parts[0] == 'oz':
                A[2,3] = num
            elif parts[0] == 'sx':
                A[0, 0] = num
            elif parts[0] == 'sy':
                A[1, 1] = num
            elif parts[0] == 'sz':
                A[2, 2] = num
            elif parts[0] == 'nl':
                # normalize diagonal with level
                np.fill_diagonal(A,A.diagonal()/(2**(num-1)))
    return A

def readParameterFile(parameterfile = ""):
    # reads calculated_parameters.txt file and parse it into a transform
    # const jobname = "ocJHfFH"
    # const nlevels = 6
    # const nchannels = 2
    # const shape_leaf_px = [406,256,152]
    # const voxelsize_used_um = [0.329714,0.342888,1.00128]
    # const origin_nm = [61677816,45421726,16585827]
    # const tile_type = convert(Cint,1)
    # const render_version = "2017-03-06 14:11:38 -0500 ec13bbfa7f9285447d3b9702b96a1f1afb847244"
    # const mltk_bary_version = "2016-11-11 12:16:03 -0500 84e153640047e3830abf835e1da4b738efa679d3"
    # const tilebase_version = "2016-08-22 15:49:39 -0400 cc171869a904e9e876426f2bb2732a38e607a102"
    # const nd_version = "2016-11-17 08:30:01 -0500 ef4923831c7bddadd0bba6b706f562a7cde00183"
    # const ndio_series_version = "2016-08-23 11:11:13 -0400 fdfe30a71f3d97fad6ac9982be50d8aea90b5234"
    # const ndio_tiff_version = "2016-08-23 11:11:54 -0400 df46d485cdf55ba66b8ed16fcf9fd9f3d5892464"
    # const ndio_hdf5_version = "2016-08-30 14:25:54 -0400 0c7ac77c5ca535913bfae5300159e6bdf60e36ca"
    # const mylib_version = "2013-08-06 19:15:35 -0400 0ca27aae55a5bab44263ad2e310e8f4507593ddc"
    params = {} # initialize dictionary
    with open(parameterfile, 'r') as f:
        while True:
            text = f.readline()
            if not text: break
            parts = text.split('=')
            keyval = parts[0].strip()
            if keyval == 'const nlevels':
                params['nlevels'] = np.array(eval(parts[1].strip('\n')),dtype=np.float)
            elif keyval == 'const shape_leaf_px':
                params['leafshape'] = np.array(eval(parts[1].strip('\n')),dtype=np.float)
            elif keyval == 'const voxelsize_used_um':
                params['vixsize'] = np.array(eval(parts[1].strip('\n')),dtype=np.float)
            elif keyval == 'const origin_nm':
                params['origin'] = np.array(eval(parts[1].strip('\n')),dtype=np.float)
            elif keyval == 'const nchannels':
                params['nchannels'] = np.array(eval(parts[1].strip('\n')),dtype=np.float)
            else:
                it=0
    A = np.zeros((3,4))
    np.fill_diagonal(A, params['vixsize']*1000) #convert to nm
    A[:,3] = params['origin']
    params['A'] = A
    return params

# ORIGINAL_SOURCE Janelia Workstation Large Volume Viewer
# OFFSET 66310.961575 46976.514329 18608.718278
# COLOR 1.000000,0.200000,0.200000
def readSWC(swcfile='./2017-06-10_G-029_Consensus.swc',scale=1.0):
    swcline=[]
    offset = np.zeros((1,3))
    offsetkey = 'OFFSET'
    header = []
    with open(swcfile, 'r') as f:
        while True:
            text = f.readline()
            if not text: break
            if text[0]=='#':
                header.append(text)
                # check offset
                if text[2:len(offsetkey)+2]==offsetkey:
                    offset = np.array(text[len(offsetkey) + 3:-1].split(), dtype=np.float).reshape(1,3)
                else:
                    continue #skip
            else:
                parts = text.split(' ')
                swcline.append(parts)
    lines = np.array(swcline, dtype=float).reshape(-1, 7)
    edges = lines[:,(0,6)]
    R = lines[:,5]
    xyz = lines[:,2:5]
    xyz = xyz + offset
    xyz = xyz/scale
    return (xyz,edges,R,offset,scale,header)

def upsampleSWC(xyz,edges,sp):
    xyzup = []
    for i, j in np.asarray(edges - 1, np.int64):
        if j < 0:
            continue
        else:
            st = xyz[i, :][None, :]
            ed = xyz[j, :][None, :]
            el = ed - st
            enel = el / np.linalg.norm(el)
            numiter = np.ceil(np.linalg.norm(el) / sp)
            xyzup.append(np.arange(0, numiter).reshape(-1, 1) * enel * sp + st)

    return(np.concatenate(xyzup))


def um2pix(um,A):
    # applies transform to convert um into pix location
    # um_ = np.concatenate((um, np.ones((um.shape[0], 1))), axis=1)
    # return(np.dot(np.linalg.pinv(A),um.T))
    return(np.dot(np.diag(1 / np.diagonal(A[:3, :3])), (um - A[:, 3]).T))

def pix2um(xyz,A):
    return(np.dot(A,xyz))
def pix2oct(xyz,dims,depth):
    # for a given xyz, box size and depth, returns the location int the patch and patch path
    res = dims/depth
    ijk = np.floor(xyz/res)
    # convert ijk to

    return 0

def um2oct(xyz,dims,transform ):
    # for a given um, transform and image size, returns the patch location
    return 0
def traverseOct():
    # lets you to traverse octree
    return 0


class Convert2JW(object):
    def __init__(self,h5file,experiment_folder,number_of_level=3):
        with h5py.File(h5file, "r") as f:
            volume = f["volume"]
            self.h5_dims= np.array(volume.shape)
            self.h5_chunk_size = np.array(volume.chunks)
            if not number_of_level:
                # estimate leaf size & depth
                # use multiple of chunk size for leaf
                self.target_leaf_size = self.h5_chunk_size[:3] * 8  # set target_leaf_size to a multiple of chunk size for efficiency
                depths = np.arange(2, 7)[:,None]
                self.output_dims = 2**depths[np.where(np.all(2**depths*self.target_leaf_size[None,:]>self.h5_dims[:3],axis=1))[0]].flatten()[0]*self.target_leaf_size
                # depths = np.arange(2, 10)[:,None]
                # self.output_dims = 2**depths[np.where(np.all(2 **depths*self.h5_chunk_size[:3][None,:]>self.h5_dims[:3],axis=1))[0]].flatten()[0]*self.h5_chunk_size[:3]
                self.number_of_level = np.log2(self.output_dims[0]/self.target_leaf_size[0]).__int__()
            else:
                self.output_dims = self.h5_dims
                self.number_of_level = number_of_level
                self.target_leaf_size = np.asarray(np.ceil(np.array(self.output_dims[:3]) / 2 ** self.number_of_level),
                                                   np.int)  # need to iter over leafs
        self.h5file = h5file
        self.experiment_folder = experiment_folder

    def __str__(self):
        return "{}\n{}\n{}\n{}\n{}".format(self.output_dims,self.number_of_level,self.target_leaf_size,self.h5file,self.experiment_folder)

    def convert2JW(self):
        h5file = self.h5file
        number_of_level = self.number_of_level
        experiment_folder = self.experiment_folder
        target_leaf_size = self.target_leaf_size
        with h5py.File(h5file, "r") as f:
            volume = f["volume"]
            output_dims = volume.shape
            bit_multiplication_array = 2**np.arange(3)
            #target_leaf_size = np.asarray(np.ceil(np.array(output_dims[:3]) / 2 ** number_of_level), np.int)
            padded_size = target_leaf_size*2**number_of_level
            range_values = [np.asarray(np.arange(0, padded_size[ii], target_leaf_size[ii]),dtype=np.int).tolist() for ii in range(3)]

            for ix,ref in enumerate(list(itertools.product(*range_values))):
                bb_end = np.asarray(np.min((ref+target_leaf_size,np.array(output_dims[:3])),axis=0),dtype=np.int)
                patch_ = volume[ref[0]:bb_end[0], ref[1]:bb_end[1], ref[2]:bb_end[2], :]
                if ~np.any(patch_):
                    continue

                # '''if patch size is smaller than full volume size pad zeros'''
                if np.any(bb_end-np.array(ref) < target_leaf_size):
                    # pad
                    patch = np.zeros(np.append(target_leaf_size,2),dtype=np.uint16)
                    patch[:patch_.shape[0], :patch_.shape[1], :patch_.shape[2],:] = patch_
                else:
                    patch = patch_

                folder_inds = np.array(np.unravel_index(ix, ([2**number_of_level for ii in range(3)])))
                folder_inds = folder_inds + 1

                patch_folder_path = []
                for im in np.arange(number_of_level,0,-1):
                    bits = folder_inds>2**(im-1)
                    # bit 2 num
                    patch_folder_path.append(1+np.sum(bits*bit_multiplication_array))
                    folder_inds = folder_inds-2**(im-1)*bits

                # create folder
                outfolder = os.path.join(experiment_folder,'/'.join(str(pp) for pp in patch_folder_path))
                # if ~np.any(patch):
                #     continue

                if not os.path.exists(outfolder):
                    os.makedirs(outfolder)
                print(outfolder)
                for ichannel in range(2):
                    outfile = os.path.join(outfolder,'default.'+str(ichannel)+'.tif')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        io.imsave(outfile, np.swapaxes(patch[:, :, :, ichannel], 2, 0))

    def create_transform_file(self):
        experiment_folder = self.experiment_folder
        number_of_level = self.number_of_level
        # create transform file
        transform_file = os.path.join(experiment_folder,'transform.txt')
        with open(transform_file, 'w') as ft:
            ft.write('ox: {:04.0f}\n'.format(0))
            ft.write('oy: {:04.0f}\n'.format(0))
            ft.write('oz: {:04.0f}\n'.format(0))
            ft.write('sx: {:.0f}\n'.format(2**number_of_level*1000))
            ft.write('sy: {:.0f}\n'.format(2**number_of_level*1000))
            ft.write('sz: {:.0f}\n'.format(2**number_of_level*1000))
            ft.write('nl: {:.0f}\n'.format(number_of_level+1))

    def create_yml_file(self):
        yml_file = os.path.join(self.experiment_folder,'tilebase.cache.yml')

    def mergeJW(self,number_of_level=3):
        # reads an octant, down samples it and save a tif file
        # create all paths from depth-1 to 0
        experiment_folder = self.experiment_folder
        leaf_size = self.target_leaf_size
        values = ['{}/'.format(str(ii + 1)) for ii in range(8)]
        for current_number_of_level in np.arange(number_of_level-1, -1, -1):
            for iter_current_folder in list(itertools.product(values, repeat=current_number_of_level)):
                my_lst_str = ''.join(map(str, iter_current_folder))
                current_folder = os.path.join(experiment_folder, my_lst_str)
                if os.path.exists(current_folder):
                    print(current_folder)
                    self.__iter_octant__(current_folder,leaf_size)

    def __iter_octant__(self,current_folder,leaf_shape):
        for ichannel in range(2):
            im_channel = []
            for ioct in range(8):
                current_path = os.path.join(current_folder,str(ioct+1))
                current_file = current_path+'/default.{}.tif'.format(ichannel)
                if os.path.exists(current_file):
                    im_batch = np.swapaxes(io.imread(current_file), 2, 0)
                else:
                    im_batch = np.zeros(leaf_shape)
                im_channel.append(im_batch)

            rt1 = np.concatenate(im_channel[0:2], axis=0)
            rt2 = np.concatenate(im_channel[2:4], axis=0)
            rt3 = np.concatenate(im_channel[4:6], axis=0)
            rt4 = np.concatenate(im_channel[6:8], axis=0)
            rt5 = np.concatenate((rt1, rt2), axis=1)
            rt6 = np.concatenate((rt3, rt4), axis=1)
            merged_Im = np.concatenate((rt5, rt6), axis=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # down sample image by 2
                down_image = resize(merged_Im, leaf_shape, preserve_range=True)
                io.imsave(current_folder+'/default.{}.tif'.format(ichannel),
                          np.asarray(np.swapaxes(down_image, 2, 0), np.uint16))

