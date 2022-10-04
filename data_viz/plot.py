from random import randint

import matplotlib.pyplot as plt

from data_access.data_paths import get_pbc_dataset_infos_paths
from data_access.data_access import get_image, load_pickle


pbc_dataset_infos_path = get_pbc_dataset_infos_paths('both')
PBC_infos_df = load_pickle(pbc_dataset_infos_path)


def all_cell_types(targets):
    """Return a figure with 2X4 axes, each of them showing a particular
    cell type with a randomly chosen representant"""

    cell_types = targets.unique()
    fig, axes = plt.subplots(2, 4)

    for cell_type, ax in zip(cell_types, axes.flatten()):

        cell_type_df = PBC_infos_df[PBC_infos_df.target ==
                                    cell_type].reset_index()
        n = len(cell_type_df)
        id_ = randint(0, n-1)

        img_path = cell_type_df.path[id_]
        img = get_image(img_path)
        ax.imshow(img)
        ax.set_title(cell_type)
        ax.axis('off')

    return fig


def cell_types_distribution(targets):
    cell_types = targets.value_counts(sort=False)
    fig, ax = plt.subplots()
    ax.bar(cell_types.index, height=cell_types.values)
    ax.tick_params(axis='x', labelrotation=45)

    return fig


def reload_content(placeholder_func, func, *args):
    placeholder_func(func(*args))
