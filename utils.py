import os
import numpy as np
import skimage  # ! skimage 0.14 not have rectangle_perimeter
import cv2
from PIL import ImageFile
import tensorflow as tf
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_img(image_path: str, input_size: tuple, resize=True)->tuple:
    """read image from the filesystem

    Parameters
    ----------
    image_path : str
        image path
    input_size : tuple
        the input size (height,width)
    resize : bool, optional
        set to resize the image (the default is True)

    Returns
    -------
    tuple
        out image source,output height, output width,input height,input width
    """
    orig_img = skimage.io.imread(image_path)
    if len(orig_img.shape) == 2:  # avoid input the gray image
        orig_img = skimage.color.gray2rgb(orig_img)
    orig_h, orig_w, _ = orig_img.shape
    if resize:
        out_img = skimage.transform.resize(orig_img, input_size, mode='reflect')
    else:
        out_img = orig_img
    out_h, out_w, _ = out_img.shape
    return out_img, out_h, out_w, orig_h, orig_w


def clean_dir(dir_name: str):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
