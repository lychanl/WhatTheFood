import argparse
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np

from whatthefood.data.xml_to_obj import parse_dir
from whatthefood.data.obj_to_nparray import get_cells, get_classes, get_output


def dir_stats(dir_name, recursive=True, visualise_grid=None):
    data = parse_dir(dir_name, recursive)

    if visualise_grid:
        visualise_grids_classes(data, visualise_grid)

    img_n = len(data)
    obj_n = sum(len(a.objects) for a in data)

    to_stat = {
        'Image dimensions': [a.img_size for a in data],
        'Objects number': [len(a.objects) for a in data],
        'Minimal bounding box distances': get_min_bb_dists(data),
        'Minimal bounding box distances (different labels)': get_min_bb_dists(data, False),
        'Bounding box centers': [o.center for a in data for o in a.objects],
        'Bounding box sizes': [o.size for a in data for o in a.objects],
        'Bounding box centers spread': get_bb_centers_ranges(data),
        'Bounding box spread': get_bb_ranges(data)
    }

    stats = {
        'Mean': lambda x: np.mean(x, axis=0),
        'Min': lambda x: np.min(x, axis=0),
        'Max': lambda x: np.max(x, axis=0),
        'Q1': lambda x: np.quantile(x, q=0.25, axis=0),
        'Q2': lambda x: np.quantile(x, q=0.5, axis=0),
        'Q3': lambda x: np.quantile(x, q=0.75, axis=0),
    }

    label_occurences = count_label_occurrences(data)

    return img_n, obj_n, {
        prop_k: {
            stat_k: stat_f(prop_v)
            for stat_k, stat_f in stats.items()
        }
        for prop_k, prop_v in to_stat.items()
    }, label_occurences


def get_min_bb_dists(data, count_same_label=True):
    return [
        np.min([
            np.max(np.abs(np.array(o1.center) - o2.center))
            for o1, o2 in itertools.combinations(a.objects, r=2) if count_same_label or o1.label != o2.label
        ], axis=0)
        for a in data
    ]


def get_bb_centers_ranges(data, count_same_label=True):
    return [
        np.max([
            (np.abs(np.array(o1.center) - o2.center))
            for o1, o2 in itertools.combinations(a.objects, r=2)
        ], axis=0)
        for a in data
    ]

def get_bb_range(center1, center2, size1, size2):
    points = [
        (np.array(center1) - size1) / 2,
        (np.array(center1) + size1) / 2,
        (np.array(center2) - size2) / 2,
        (np.array(center2) + size2) / 2
    ]
    return np.max([
        np.abs(p1 - p2)
        for p1, p2 in itertools.combinations(points, r=2)
    ], axis=0)


def get_bb_ranges(data):
    return [
        np.max([
            get_bb_range(o1.center, o2.center, o1.size, o2.size)
            for o1, o2 in itertools.combinations(a.objects, r=2)
        ], axis=0)
        for a in data
    ]


def count_label_occurrences(data):
    occ = {}

    for a in data:
        labels = [o.label for o in a.objects]

        for label in labels:
            if label in occ:
                occ[label][0] += 1
            else:
                occ[label] = [1, 0]
        for label in set(labels):
            occ[label][1] += 1

    return occ


def visualise_grids_classes(data, n_cells):
    classes = get_classes(data)

    outputs = [get_output(a, classes, n_cells) for a in data]
    outs_sum = sum(outputs)
    outs_sum = outs_sum + np.flip(outs_sum, 0) + np.flip(outs_sum, 1) + np.flip(outs_sum, (0, 1))

    # log color scale
    outs_sum = np.log(outs_sum + 1)

    nrows = 3

    fig, _axs = plt.subplots(nrows, len(classes) // nrows + 1)

    def get_axis(i):
        return _axs[i % nrows, i // nrows]

    m = get_axis(0).matshow(outs_sum[:, :, 0])
    plt.colorbar(m, ax=get_axis(0))
    get_axis(0).set_title('ANY')

    for i, c in enumerate(classes, start=1):
        m = get_axis(i).matshow(outs_sum[:, :, 4 + i])
        plt.colorbar(m, ax=get_axis(i))
        get_axis(i).set_title(c)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_name', type=str)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--visualise-grid-classes', type=int, default=None)

    args = parser.parse_args()

    img_n, obj_n, stats, lab_occ = dir_stats(
        args.dir_name,
        recursive=args.recursive,
        visualise_grid=args.visualise_grid_classes
    )

    print(f'{img_n} images')
    print(f'{obj_n} objects')
    print()

    print("Statistics:")
    print()

    for prop_name, prop_stats in stats.items():
        print(f'{prop_name}:')
        for stat_name, stat_value in prop_stats.items():
            print(f'\t{stat_name}: {stat_value}')
        print()

    print('Labels occurrences (all, in distinct images):')

    for lab, (occ, distinct) in lab_occ.items():
        print(f'\t{lab}: {occ}, {distinct}')
