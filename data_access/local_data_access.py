import pickle
from PIL import Image


def get_image(img_path):
    """Return the image located at img_path on the local storage."""
    return Image.open(img_path)


def get_dataset_infos():
    """Return a DataFrame containing path, width, height and cell's type for each image."""
    with open('data_access/PBC_infos.PICKLE', 'rb') as f:
        PBC_infos_df = pickle.load(f)
    return PBC_infos_df