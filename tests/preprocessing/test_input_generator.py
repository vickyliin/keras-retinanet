from glob import glob
import pytest
from skimage.io import imread
from keras_retinanet.preprocessing import InputGenerator

def test_input_generator():
    image_names = glob('tests/test-data/csv/images/*')
    img_org = imread(image_names[0])
    ig = InputGenerator(image_names, 
                        batch_size=1, 
                        image_min_side=300, 
                        image_max_side=1024)
    img, (scale, path) = ig.next()
    assert min(img.shape[1:3]) == 300
    assert scale == min(img.shape[1:3]) / min(img_org.shape[0:2])
    assert path == image_names[0]
