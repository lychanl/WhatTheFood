from whatthefood.data.xml_to_obj import parse_dir
from whatthefood.data.obj_to_nparray import get_dataset
from whatthefood.data.preprocessing import ScalePreprocessor

import argparse
import pickle


def create_dataset(dirname, outname, hcells, wcells=None, scale=13):
    anns = parse_dir(dirname)
    preprocessing = ScalePreprocessor(13)
    data = get_dataset(anns, hcells, wcells, preprocessing)

    with open(outname, 'wb') as outfile:
        pickle.dump(data, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('out', type=str)

    parser.add_argument('--hcells', type=int, required=True)
    parser.add_argument('--wcells', type=int, required=False, default=None)

    parser.add_argument('--scale', type=int, required=False, default=13)

    args = parser.parse_args()

    create_dataset(args.dir, args.out, args.hcells, args.wcells, args.scale)
