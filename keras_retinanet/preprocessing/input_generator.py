import os.path
import numpy as np
from .generator import Generator
from ..utils.image import read_image_bgr

class InputGenerator(Generator):
    def __init__(self, image_names, **kwargs):
        self.image_names = np.array(image_names)
        super().__init__(shuffle_groups=False, 
                         group_method='none',
                         **kwargs)

    def load_image(self, image_index):
        path = self.image_path(image_index)
        return path, read_image_bgr(path)

    def image_path(self, image_index):
        return self.image_names[image_index]

    def __iter__(self):
        return self.iterate_once()

    def iterate_once(self):
        return super().iterate_once(get_annotations=False)
