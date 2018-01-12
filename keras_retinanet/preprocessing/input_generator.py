import os.path
from .generator import Generator
from ..utils.image import read_image_bgr

class InputGenerator(Generator):
    def __init__(self, image_names, base_dir=None, **kwargs):
        self.image_names = image_names
        self.base_dir = base_dir or ''
        super().__init__(**kwargs)

    def size(self):
        return len(self.image_names)

    def load_image(self, image_index):
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        return None

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def preprocess_group(self, image_group):
        for index, image in enumerate(image_group):
            image = image_group[index] = self.preprocess_image(image)
            image_group[index] = self.resize_image(image)[0]
        return image_group

    def compute_input_output(self, group):
        image_group = self.load_image_group(group)
        image_group = self.preprocess_group(image_group)
        return self.compute_inputs(image_group)
