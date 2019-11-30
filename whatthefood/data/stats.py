import itertools
import sys

import numpy as np

from whatthefood.data.xml_to_obj import parse_dir


def dir_stats(dir_name):
    data = parse_dir(dir_name)

    n = len(data)

    to_stat = {
        'Image dimensions': [a.img_size for a in data],
        'Objects number': [len(a.objects) for a in data],
        'Minimal bounding box distances': get_min_bb_dists(data),
        'Minimal bounding box distances (different labels)': get_min_bb_dists(data, False),
        'Bounding box centers': [o.center for a in data for o in a.objects],
        'Bounding box sizes': [o.size for a in data for o in a.objects],
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

    return n, {
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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid number of arguments 1 expected:")
        print(" - directory name")
        exit(1)

    n, stats, lab_occ = dir_stats(sys.argv[1])

    print(f'{n} images')
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
