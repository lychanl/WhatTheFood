import numpy as np
from collections.abc import Iterable

from PIL import Image

from whatthefood.data.dataset import Dataset
from whatthefood.data.xml_to_obj import Object


def get_dataset(annotations, n_cells_h, n_cells_w=None, preprocessing=None):
    classes = get_classes(annotations)

    if not n_cells_w:
        n_cells_w = n_cells_h

    names = []
    inputs = []
    outputs = []

    for a in annotations:
        names.append(a.img_name)
        inputs.append(load_input_image(a.img_path, preprocessing))
        outputs.append(get_output(a, classes, n_cells_h, n_cells_w))

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return Dataset(names, inputs, outputs, classes)


def get_classes(annotations):
    return list(sorted(set(o.label for a in annotations for o in a.objects)))


def _get_cells(height, width, hcells, wcells):
    cell_size = min(width / wcells, height / hcells)
    w_offset = (width - wcells * cell_size) / 2
    h_offset = (height - hcells * cell_size) / 2

    cells = ([], [])

    y = h_offset
    for j in range(hcells):
        y += cell_size
        if j == hcells - 1:
            y = height

        cells[0].append(y)

    x = w_offset
    for i in range(wcells):
        x += cell_size
        if i == wcells - 1:
            x = width

        cells[1].append(x)

    return cells


def get_cells(img_size, hcells, wcells):
    size = img_size

    return _get_cells(*size[:2], hcells, wcells)


def get_cell(y, x, cells):
    i, j = 0, 0

    for i in range(len(cells[0])):
        if i == len(cells[0]) - 1 or cells[0][i] > y:
            break

    for j in range(len(cells[1])):
        if j == len(cells[1]) - 1 or cells[1][j] > x:
            break

    return i, j


def get_output(annotation, classes, n_cells_h, n_cells_w):
    cells = get_cells(annotation.img_size, n_cells_h, n_cells_w)

    out = np.zeros((len(cells[0]), len(cells[1]), 5 + len(classes)))

    for o in annotation.objects:
        i, j = get_cell(*o.center, cells)

        y = o.center[0] / cells[0][0]\
            if i == 0 else\
            (o.center[0] - cells[0][i - 1]) / (cells[0][i] - cells[0][i - 1])
        x = o.center[1] / cells[1][0]\
            if j == 0 else\
            (o.center[1] - cells[1][j - 1]) / (cells[1][j] - cells[1][j - 1])

        h = o.size[0] / annotation.img_size[0]
        w = o.size[1] / annotation.img_size[1]

        if out[(i, j, 0)] == 1:
            raise ValueError(f'More then 1 object in cell for image {annotation.img_name}')
        out[(i, j)] = [1., y, x, h, w] + [1. if o.label == c else 0. for c in classes]

    return out


def get_objects_from_output(output, img_size, classes, ncells):
    # cells = get_cells(img_size[:2], ncells, ncells)
    cells = get_cells(img_size[:2], output.shape[-3], output.shape[-2])

    def cell_size(d, i):
        return cells[d][0] if i == 0 else cells[d][i] - cells[d][i - 1]

    def cell_height(i):
        return cell_size(0, i)

    def cell_width(i):
        return cell_size(1, i)

    def y(i, cell_y):
        return cell_height(i) * cell_y + (0 if i == 0 else cells[0][i - 1])

    def x(j, cell_x):
        return cell_width(j) * cell_x + (0 if j == 0 else cells[1][j - 1])

    def out_to_obj(cell_out, i, j):
        o = Object()
        o.center = (y(i, cell_out[1]), x(j, cell_out[2]))
        o.size = (img_size[0] * cell_out[3], img_size[1] * cell_out[4])
        o.label = classes[np.argmax(cell_out[5:])]

        return o

    return [
        out_to_obj(cell, i, j)
        for i, col in enumerate(output) for j, cell in enumerate(col) if cell[0] > 0.5
    ]


def load_input_image(path, preprocessing=None):
    with Image.open(path) as img:
        img_array = np.array(img, dtype=np.float32) / 255

    if preprocessing:
        if isinstance(preprocessing, Iterable):
            for p in preprocessing:
                img_array = p(img_array)
        else:
            img_array = preprocessing(img_array)

    return img_array
