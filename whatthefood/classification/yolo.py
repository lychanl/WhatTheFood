import whatthefood.graph as graph


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
