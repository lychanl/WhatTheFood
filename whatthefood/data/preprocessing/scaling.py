import numpy as np

from whatthefood.data.preprocessing.preprocessor import Preprocessor


class ScalePreprocessor(Preprocessor):
    def __init__(self, scale, fun=np.mean):
        self.scale = scale
        self.fun = fun

    def __call__(self, obj):
        return self.fun(
            obj.reshape(
                obj.shape[0] // self.scale,
                self.scale,
                obj.shape[1] // self.scale,
                self.scale,
                obj.shape[2]
            ),
            axis=(1, 3)
        )
