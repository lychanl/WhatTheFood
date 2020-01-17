from collections.abc import Iterable
import numpy as np


def load_tensorflow():
    import tensorflow
    tensorflow.compat.v1.disable_eager_execution()
    return tensorflow


def run(ops, placeholders_dict=None, tf_sess=None):
    if placeholders_dict is None:
        placeholders_dict = {}
    if tf_sess:
        return _run_tf(ops, placeholders_dict, tf_sess)
    else:
        return _run_no_tf(ops, placeholders_dict)


def _run_no_tf(ops, placeholders_dict):
    if isinstance(ops, Iterable):
        to_do = list(ops)
    else:
        to_do = [ops]

    values = {}
    for placeholder, value in placeholders_dict.items():
        values[placeholder] = value if isinstance(value, np.ndarray) else np.array(value)

    while to_do:
        op = to_do[-1]
        if op in values:
            to_do.pop()
        else:
            missing = False

            for inp in op.inputs:
                if inp not in values:
                    missing = True
                    to_do.append(inp)

            if not missing:
                input_values = [values[i] for i in op.inputs]
                output = op.do(*input_values)
                values[op] = output

                to_do.pop()

    if isinstance(ops, Iterable):
        return [values[o] for o in ops]
    else:
        return values[ops]


def _run_tf(ops, placeholder_dict, sess):
    tf = load_tensorflow()

    tf_ops = [o.build_tf(tf, sess) for o in ops] if isinstance(ops, Iterable) else ops.build_tf(tf, sess)
    tf_placeholder_dict = {p.build_tf(tf, sess): v for p, v in placeholder_dict.items()}

    return sess.run(tf_ops, tf_placeholder_dict)

