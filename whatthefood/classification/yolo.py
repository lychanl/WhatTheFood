import whatthefood.graph as graph
from whatthefood.nn import Model, mse, layers
from whatthefood.classification.utils import to_classes
from whatthefood.classification.metrics import accuracy, tp_rate, fp_rate

import numpy as np


def split_output(output):
    existence = graph.Slice(output, (0, 0, 0), (*output.shape[:2], 1))
    bounding_boxes = graph.Slice(output, (0, 0, 1), (*output.shape[:2], 5))
    classes = graph.Slice(output, (0, 0, 5), output.shape)

    return existence, bounding_boxes, classes


def yolo_activation(output):
    assert len(output.shape) == 3
    assert output.shape[2] > 5

    existence, bounding_boxes, classes = split_output(output)

    existence = graph.Sigmoid(existence)
    classes = graph.Softmax(classes)

    return graph.Concatenate((existence, bounding_boxes, classes), axis=-1)


def yolo_loss(expected, result, noobj_weight=.5, bounding_boxes_weight=5):
    assert len(expected.shape) == 3
    assert expected.shape[2] > 5

    assert expected.shape == result.shape
    assert expected.batched == result.batched

    e_existence, e_bounding_boxes, e_classes = split_output(expected)
    existence, bounding_boxes, classes = split_output(result)

    ow = graph.Constant(1 - noobj_weight)
    ew = graph.Constant(noobj_weight)
    bbw = graph.Constant(bounding_boxes_weight)

    existence_loss = graph.ReduceSum(
        graph.Square(graph.Difference(e_existence, existence)),
        axis=-1, reduce_batching=False)
    bounding_boxes_loss = graph.ReduceSum(
        graph.Square(graph.Difference(e_bounding_boxes, bounding_boxes)),
        axis=-1, reduce_batching=False)
    classes_loss = graph.ReduceSum(
        graph.Square(graph.Difference(e_bounding_boxes, bounding_boxes)),
        axis=-1, reduce_batching=False)

    weighted_existence_loss = graph.MultiplyByScalar(existence_loss, ew)
    weighted_obj_loss = graph.MultiplyByScalar(existence_loss, ow)
    weighted_bounding_boxes_loss = graph.MultiplyByScalar(bounding_boxes_loss, bbw)

    total_obj_loss = graph.Sum(graph.Sum(weighted_obj_loss, weighted_bounding_boxes_loss), classes_loss)
    total_loss = graph.Sum(
        graph.Multiply(total_obj_loss, graph.Reshape(e_existence, e_existence.shape[:2])),
        weighted_existence_loss)

    return graph.ReduceMean(graph.ReduceSum(total_loss, reduce_batching=False))


def yolo_metrics(expected, result):
    e_existence, e_bounding_boxes, e_classes = split_output(expected)
    existence, bounding_boxes, classes = split_output(result)

    existence = to_classes(existence)
    classes = to_classes(classes)

    e_existence = to_classes(e_existence)
    e_classes = to_classes(e_classes)

    flags = e_existence

    existence_accuracy = accuracy(e_existence, existence)
    existence_tp_rate = tp_rate(e_existence, existence)
    existence_fp_rate = fp_rate(e_existence, existence)
    bb_error = mse(e_bounding_boxes, bounding_boxes)
    classes_accuracy = accuracy(e_classes, classes, flags=flags)

    return existence_accuracy, existence_tp_rate, existence_fp_rate, bb_error, classes_accuracy


def lenet_7_yolo_net(input_shape, output_shape):
    model = Model()
    model.add(graph.Placeholder, input_shape, batched=True)
    model.add(layers.Convolution, 30, 5, 1, activation=graph.ReLU)
    model.add(graph.MaxPooling2d, 4)
    model.add(layers.Convolution, 90, 5, 1, activation=graph.ReLU)
    model.add(graph.MaxPooling2d, 3)
    model.add(layers.Convolution, 360, 6, 1, activation=graph.ReLU)
    model.add(layers.Convolution, 60, 1, 1, activation=graph.ReLU)
    model.add(graph.flatten)
    model.add(layers.Dense, 1024, activation=graph.ReLU)
    model.add(layers.Dense, np.prod(output_shape))
    model.add(graph.Reshape, output_shape)
    model.add(yolo_activation)

    return model
