import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse

from whatthefood.data.xml_to_obj import parse_file
from whatthefood.data.obj_to_nparray import get_objects_from_output, load_input_image
from whatthefood.data.preprocessing import ScalePreprocessor
from whatthefood.classification.utils import get_output_mean_with_flipped


def visualise_objects(ax, objects, color, scale=None):
    for o in objects:
        loc = (o.center[1] - o.size[1] / 2, o.center[0] - o.size[0] / 2)
        size = (o.size[1], o.size[0])
        if scale:
            loc = (loc[0] / scale, loc[1] / scale)
            size = (size[0] / scale, size[1] / scale)
        ax.add_patch(plt.Rectangle(loc, size[0], size[1], fill=False, linewidth=2, edgecolor=color))
        ax.text(loc[0], loc[1], o.label,
                color=color, weight="bold",
                verticalalignment="bottom",
                horizontalalignment="left")


def visualise_img_and_annot(img, objects, annot_objects, scale=None):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

    if annot_objects:
        print(f"Actual objects number: {len(annot_objects)}")
        visualise_objects(ax, annot_objects, 'green', scale)
    if objects:
        print(f"Detected objects number: {len(objects)}")
        visualise_objects(ax, objects, 'red')

    plt.show()


def visualise_data(img, expected_out, model_out, classes):
    yolo_out_objs = get_objects_from_output(model_out, img.shape[:2], classes) if model_out is not None else None
    yolo_exp_objs = get_objects_from_output(expected_out, img.shape[:2], classes) if expected_out is not None else None

    visualise_img_and_annot(img, yolo_out_objs, yolo_exp_objs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', action='store', type=str, default=None, required=False)
    parser.add_argument('--model-classes-from', action='store', type=str, default=None, required=False)
    parser.add_argument('--annotation', action='store_const', const=True, default=False)
    parser.add_argument('img')

    args = parser.parse_args()

    flips = False

    model = None
    preprocessor = None
    scale = None
    if args.model:
        with open(args.model, 'rb') as file:
            model = pickle.load(file)
        scale = 2340 // model.inputs[0].shape[0]
        assert scale == 4160 // model.inputs[0].shape[1]
        preprocessor = ScalePreprocessor(scale, np.mean)

    if not args.annotation:
        annot = None
        img = load_input_image(args.img, preprocessor)
    else:
        annot = parse_file(args.img)
        img = load_input_image(annot.img_path, preprocessor)

    yolo_out_annot = None

    if model:
        ds = None
        if args.model_classes_from:
            with open(args.model_classes_from, 'rb') as file:
                ds = pickle.load(file)

        classes = list(range(model.output.shape[2] - 5)) if not ds else ds.classes
        if flips:
            out = get_output_mean_with_flipped(model, [img])[0]
        else:
            out = model([img])[0]

        yolo_out_annot = get_objects_from_output(out, img.shape[:2], classes)

    visualise_img_and_annot(img, yolo_out_annot, annot.objects, scale)
