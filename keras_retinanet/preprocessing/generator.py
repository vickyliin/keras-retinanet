"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import random
import threading
import time
import warnings
from argparse import Namespace

import keras
from keras.preprocessing.image import ImageDataGenerator

from ..utils.image import preprocess_image, resize_image, random_transform
from ..utils.anchors import anchor_targets_bbox


class Generator(object):
    def __init__(
        self,
        batch_size,
        image_min_side,
        image_max_side,
        group_method='random',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        thresh=0,
        seed=None
    ):
        self.batch_size           = int(batch_size)
        self.group_method         = group_method
        self.shuffle_groups       = shuffle_groups
        self.image_min_side       = image_min_side
        self.image_max_side       = image_max_side
        self.thresh               = thresh

        if seed is None:
            seed = np.uint32((time.time() % 1)) * 1000
        np.random.seed(seed)

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group.org, annotations_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def preprocess_image_group(self, image_group):
        image_group = Namespace(
            org=image_group,
            path=[],
            resized=[],
            scale=[],
            input=[]
        )
        for index, (path, org) in enumerate(image_group.org):
            # resize image
            resized, scale = self.resize_image(org)

            # preprocess the image (subtract imagenet mean)
            input = self.preprocess_image(resized)

            # copy processed data back to group
            image_group.org[index] = org
            image_group.path.append(path)
            image_group.resized.append(resized)
            image_group.scale.append(scale)
            image_group.input.append(input)

        return image_group

    def preprocess_annotations_group(self, image_group, annotations_group):
        for index, (scale, annotations) in enumerate(zip(image_group.scale, annotations_group)):
            # apply resizing to annotations too
            annotations[:, :4] *= scale

            # copy processed data back to group
            annotations_group[index] = annotations

        return annotations_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        groups = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        self.groups = np.array(groups, dtype=np.object)
        self.indices = list(range(len(self.groups)))

    def compute_inputs(self, group):
        image_group = self.load_image_group(group)
        image_group = self.preprocess_image_group(image_group)
        batch_size = len(group)

        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group.input) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group.input):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        image_group.input = image_batch

        return image_group

    def anchor_targets(
        self,
        image_shape,
        boxes,
        num_classes,
        mask_shape=None,
        negative_overlap=0.4,
        positive_overlap=0.5,
        **kwargs
    ):
        return anchor_targets_bbox(image_shape, boxes, num_classes, mask_shape, negative_overlap, positive_overlap, **kwargs)

    def compute_targets(self, group, image_group):
        batch_size = len(group)
        annotations_group = self.load_annotations_group(group)
        annotations_group = self.preprocess_annotations_group(image_group, annotations_group)
        annotations_group = Namespace(annotations=annotations_group)

        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group.input) for x in range(3))

        # compute labels and regression targets
        labels_group     = [None] * batch_size
        regression_group = [None] * batch_size
        for index, (image, annotations) in enumerate(zip(image_group.input, annotations_group.annotations)):

            ignore_box_ids = np.where(annotations[..., 5] < self.thresh)
            labels_group[index], regression_group[index] = self.anchor_targets(max_shape, annotations, self.num_classes(), 
                                                                               mask_shape=image.shape, 
                                                                               ignore_box_ids=ignore_box_ids)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states           = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch     = np.zeros((batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression

        annotations_group.regression = regression_batch
        annotations_group.labels = labels_batch

        return annotations_group

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.indices)
            group = self.groups[self.indices[self.group_index]]
            self.group_index = (self.group_index + 1) % len(self.groups)

        image_group = self.compute_inputs(group)
        annotations_group = self.compute_targets(group, image_group)
        inputs = image_group.input
        targets = [annotations_group.regression, annotations_group.labels]
        return inputs, targets

    def iterate_once(self, get_annotations=True):
        for group in self.groups:
            image_group = self.compute_inputs(group)
            if get_annotations:
                annotations_group = self.compute_targets(group, image_group)
                yield image_group, annotations_group
            else:
                yield image_group
