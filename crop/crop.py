from skimage.filters import threshold_otsu
import numpy as np
from PIL import Image


def get_limits(img, axis, size, ratio=1.25):
    """Return the coordinate along the specified axis where to crop the image and
    the threshold used to find them,  augmented with the specified ratio.

    For example, if axis is 0, it will return left and right border of the zone
    of the image wthat needs to be kept after crop.

    The threshold is automaticaly computed and may be augmented to avoid potentials
    peripherical outliers to be taken in consideration."""
    std_img = img.std(axis=axis)
    t = threshold_otsu(std_img)

    t_std_img = std_img > t * ratio

    j_min = np.argmax(t_std_img)
    j_max = size - np.argmax(t_std_img[::-1])
    return j_min, j_max, t*ratio


def get_corrected_limits(u, d, l, r, w, hp):
    pass


def crop_pil(img: Image.Image, p: int = 20, output_color: str = "color"):
    """Automatically crop the image (as a PIL image) around the cell."""
    img_c = np.asarray(img)

    img_gs = np.asarray(img.convert('L'))
    height, width = img_gs.shape

    j_left, j_right, _ = get_limits(img_gs, 0, width)
    j_up, j_down, _ = get_limits(img_gs, 1, height)

    pw = width // p
    ph = height // p
    up = max(j_up - ph, 0)
    down = min(j_down + ph, height)
    left = max(j_left - pw, 0)
    right = min(j_right + pw, width)

    if output_color == 'color':
        img_ = np.copy(img_c)
    else:
        img_ = np.copy(img_gs)

    for i in range(height):
        if i < up or i > down:
            img_[i] = 0

    for j in range(width):
        if j < left or j > right:
            img_[:, j] = 0

    return Image.fromarray(img_)


def crop_np(img_: np.ndarray, p: int = 20):
    """Automatically crop the image (as a numpy array) around the cell."""
    img = img_.copy()
    height, width = img.shape[:2]

    j_left, j_right, _ = get_limits(img, 0, width)
    j_up, j_down, _ = get_limits(img, 1, height)

    pw = width // p
    ph = height // p
    up = max(j_up - ph, 0)
    down = min(j_down + ph, height)
    left = max(j_left - pw, 0)
    right = min(j_right + pw, width)

    for i in range(height):
        if i < up or i > down:
            img[i] = 0

    for j in range(width):
        if j < left or j > right:
            img[:, j] = 0
    return img
