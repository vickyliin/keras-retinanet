import os.path
import numpy as np
from .generator import Generator
from ..utils.image import read_image_bgr

class InputGenerator(Generator):
    def __init__(self, image_names, **kwargs):
        self.image_names = np.array(image_names)
        super().__init__(shuffle_groups=False, 
                         group_method='none',
                         image_data_generator=None, 
                         **kwargs)

    def load_image(self, image_index):
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        return np.ones([1, 1])

    def image_path(self, image_index):
        return self.image_names[image_index]

    def filter_annotations(self, image_group, annotations_group, group):
        return image_group, annotations_group

    def compute_targets(self, image_group, annotations_group):
        return np.concatenate(annotations_group, axis=0).squeeze(1)

    def compute_input_output(self, group):
        paths = self.image_names[group]
        inputs, scales = super().compute_input_output(group)
        return inputs, paths, scales
