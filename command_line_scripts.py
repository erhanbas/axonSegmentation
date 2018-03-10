# -*- coding: utf-8 -*-
"""Creates a shell script to run batch process on command line."""

import os
import sys, getopt


def main(argv):
    # parses input repo to generate list of consensus swcs
    function_path = '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/axonSegmentation/segmentAxon.py'
    input_folder =  '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo'
    output_folder = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo'
    output_sh_file = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/run_segmentor.sh'

    folds = [os.path.join(input_folder, fold) for fold in os.listdir(input_folder) if
               os.path.isdir(os.path.join(input_folder, fold))]
    for fold in folds:
        try:
            os.chmod(fold, 0o770)
        except:
            print(fold)

    # consensus_swc_files = [os.path.join(root, name) for root, dirs, files in os.walk(input_folder) for name in files if name.endswith(("Consensus.swc", "consensus.swc"))]
    h5files = [os.path.join(input_folder,fold,files) for fold in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, fold)) for files in os.listdir(os.path.join(input_folder,fold)) if files.endswith("onsensus-carved.h5")]
    h5files.sort()

    # for each consensus file run navigator script
    # usage:
    # print('navigator.py -i <data_folder> -s <swc_file> -o <output_folder>')

    with open(output_sh_file,'w') as fswc:
        for h5_file in h5files:
            path, filename = os.path.split(h5_file)
            output_folder = path
            # check if there is transform file in target folder
            # if os.path.exists(os.path.join(output_folder, filename.split('.')[0],'JW','transform.txt')):
            mystr = 'python {} -i {} -o {} > {} &\n'.format(function_path,
                                                            h5_file,
                                                            output_folder,
                                                            os.path.join(output_folder, 'segment_log.txt'))
            fswc.write(mystr)

    os.system('chmod +x '+output_sh_file)

if __name__ == "__main__":
   main(sys.argv[1:])

# ##
# plt.figure()
# # plt.imshow(np.max(filtresponse**.05,axis=2).T)
# plt.imshow(scaleresponse[:,:,12].T)
# plt.plot(path_array[:,0],path_array[:,1])
#
#
# sitk.Show(sitk.GetImageFromArray(np.swapaxes(segment.mask_ActiveContour,2,0)))
# sitk.Show(sitk.GetImageFromArray(np.swapaxes(self.mask_Threshold,2,0)))
#
# sitk.Show(sitk.GetImageFromArray(np.swapaxes(inputim,2,0)))
# sitk.Show(sitk.GetImageFromArray(np.swapaxes(filtresponse/np.max(filtresponse),2,0)))
# sitk.Show(sitk.GetImageFromArray(np.swapaxes(scaleresponse,2,0)))
#
# inim = segment.convert2itk(filtresponse)  # converts to itk u8bit image
# cost = sitk.GetImageFromArray(np.swapaxes(self.cost_array, 2, 0))
# cc = sitk.GetArrayFromImage(cost)
# # if ~np.all(np.isnan(self.cost_array)):
# #     cost = self.convert2itk(self.cost_array)
# # else:
# #     cost = self.cost_array
# seg = sitk.Image(inim.GetSize(), sitk.sitkUInt8)  # holder for initialization
# seg.CopyInformation(inim)
# for idx, seed in enumerate(self.seeds):
#     seg[seed.tolist()] = 1
# # Binary dilate enlarges the seed mask by 3 pixels in all directions.
# seg = sitk.BinaryDilate(seg, radius)
# # based on thresholding
# self.mask_Threshold = np.swapaxes(sitk.GetArrayFromImage(self.segmentBasedOnThreshold(inim, seg)), 2, 0)
# # based on active contours
# self.mask_ActiveContour = np.swapaxes(sitk.GetArrayFromImage(self.segmentBasedOnActiveContours(inim, seg, cost)), 2, 0)
