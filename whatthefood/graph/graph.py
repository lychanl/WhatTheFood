from collections.abc import Iterable
import numpy as np


def run(ops, placeholders_dict=None):
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
