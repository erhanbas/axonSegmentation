# -*- coding: utf-8 -*-
"""Creates a shell script to run batch process on command line."""

import os


def main(argv):
    # parses input repo to generate list of consensus swcs
    function_path = '/groups/mousebrainmicro/home/base/CODE/MOUSELIGHT/axonSegmentation/segmentAxon.py'
    input_folder =  '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo'
    output_folder = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo'
    output_sh_file = '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/run_segmentor.sh'

    folds = [os.path.join(input_folder, fold, 'JW') for fold in os.listdir(input_folder) if
               os.path.isdir(os.path.join(input_folder, fold))]

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

    os.system('chmod g+x '+output_sh_file)
