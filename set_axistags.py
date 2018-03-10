import h5py
import sys
import json
def main():
    1


if __name__ == "__main__":
    main(sys.argv[1:])

    with open('/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/axis_tags_raw.json','r') as f:
       raw_tag_data = json.load(f)

    raw_render={"axes":
            [
                {
                    "key": "t",
                    "typeFlags": 8,
                    "resolution": 0,
                    "description": ""
                },
                {
                    "key": "c",
                    "typeFlags": 1,
                    "resolution": 0,
                    "description": ""
                },
                {
                  "key": "z",
                  "typeFlags": 2,
                  "resolution": 0,
                  "description": ""
                },
                {
                  "key": "y",
                  "typeFlags": 2,
                  "resolution": 0,
                  "description": ""
                },
                {
                  "key": "x",
                  "typeFlags": 2,
                  "resolution": 0,
                  "description": ""
                }
        ]
    }

    with h5py.File('/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/test_samples/183588.h5', 'r') as render_data_h5:
        rt=render_data_h5['data'][:]
    with h5py.File(
            '/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/test_samples/183588-2.h5',
            'w') as render_data_h5:
        rt=rt.reshape(rt.shape+(1,))
        rt=rt.swapaxes(0,4)

        dset_trace = render_data_h5.create_dataset("data", data=rt,
                                                dtype='uint16',
                                                compression="gzip",
                                                compression_opts=9)
        render_data_h5['data'].attrs['axistags'] = raw_tag_data.__str__()


    with h5py.File('/groups/mousebrainmicro/mousebrainmicro/users/base/AnnotationData/h5repo/2017-09-25_G-007_consensus/183588.h5', 'r+') as render_data_h5:
        render_data_h5['data'].attrs['axistags'] = raw_render.__str__()

