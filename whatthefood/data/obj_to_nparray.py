import numpy as np
from collections.abc import Iterable

from PIL import Image

from whatthefood.data.dataset import Dataset


def get_dataset(annotations, n_cells, preprocessing=None):
    classes = get_classes(annotations)

    names = []
    inputs = []
    outputs = []

    for a in annotations:
        names.append(a.img_name)
        inputs.append(load_input_image(a.img_path, preprocessing))
        outputs.append(get_output(a, classes, n_cells))

    return Dataset(names, inputs, outputs, classes)


def get_classes(annotations):
    return list(set(o.label for a in annotations for o in a.objects))


def _get_cells(width, height, wcells, hcells):
    cell_size = min(width / wcells, height / hcells)
    w_offset = (width - wcells * cell_size) / 2
    h_offset = (height - hcells * cell_size) / 2

    cells = ([], [])

    x = w_offset
    for i in range(wcells):
        x += cell_size
        if i == wcells - 1:
            x = width

        cells[0].append(x)

    y = h_offset
    for j in range(hcells):
        y += cell_size
        if j == hcells - 1:
            y = height

        cells[1].append(y)

    return cells


def get_cells(annotation, wcells, hcells):
    size = annotation.img_size

    return _get_cells(*size[:2], wcells, hcells)


def get_cell(x, y, cells):
    i, j = 0, 0

    for i in range(len(cells[0])):
        if i == len(cells[0]) - 1 or cells[0][i] > x:
            break

    for j in range(len(cells[1])):
        if j == len(cells[1]) - 1 or cells[1][j] > y:
            break

    return i, j


def get_output(annotation, classes, n_cells):
    cells = get_cells(annotation, n_cells, n_cells)

    out = np.zeros((len(cells[0]), len(cells[1]), 5 + len(classes)))

    for o in annotation.objects:
        i, j = get_cell(*o.center, cells)

        x = o.center[0] / cells[0][0]\
            if i == 0 else\
            (o.center[0] - cells[0][i - 1]) / (cells[0][i] - cells[0][i - 1])
        y = o.center[1] / cells[1][0]\
            if j == 0 else\
            (o.center[1] - cells[1][j - 1]) / (cells[1][j] - cells[1][j - 1])

        w = o.size[0] / annotation.img_size[0]
        h = o.size[1] / annotation.img_size[1]

        if out[(i, j, 0)] == 1:
            raise ValueError(f'More then 1 object in cell for image {annotation.img_name}')
        out[(i, j)] = [1., x, y, w, h] + [1. if o.label == c else 0. for c in classes]

    return out


def load_input_image(path, preprocessing=None):
    with Image.open(path) as img:
        img_array = np.array(img) / 255

    if preprocessing:
        if isinstance(preprocessing, Iterable):
            for p in preprocessing:
                img_array = p(img_array)
        else:
            img_array = preprocessing(img_array)

    return img_array
