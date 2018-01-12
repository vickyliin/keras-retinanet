import os.path
import numpy as np
from .generator import Generator
from ..utils.image import read_image_bgr

class InputGenerator(Generator):
    def __init__(self, image_names, base_dir=None, **kwargs):
        self.image_names = image_names
        self.base_dir = base_dir or ''
        super().__init__(shuffle_groups=False, 
                         group_method='none',
                         image_data_generator=None, 
                         **kwargs)

    def load_image(self, image_index):
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        return np.ones([1, 1])

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def filter_annotations(self, image_group, annotations_group, group):
        return image_group, annotations_group

    def compute_targets(self, image_group, annotations_group):
        return np.concatenate(annotations_group, axis=0).squeeze(1)
