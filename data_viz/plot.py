from random import randint
import numpy as np
import matplotlib.pyplot as plt
from data_access.data_access import get_dataset_infos, get_image, load_pickle_data, load_pickle_selector


PBC_infos_df = get_dataset_infos()


def all_cell_types(targets):
    """Return a figure with 2X4 axes, each of them showing a particular
    cell type with a randomly choose representant"""

    cell_types = targets.unique()
    fig, axes = plt.subplots(2, 4)

    for cell_type, ax in zip(cell_types, axes.flatten()):

        cell_type_df = PBC_infos_df[PBC_infos_df.cell_type ==
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


def plot_color_and_bg_img():

    random_id = randint(0, len(PBC_infos_df))
    img_path, cell_type = PBC_infos_df[["path", "cell_type"]].iloc[random_id]

    img = get_image(img_path)
    img_bg = img.convert('L')

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(cell_type)
    axes[0].imshow(img)
    axes[0].set_axis_off()
    axes[1].imshow(img_bg, cmap='gray')
    axes[1].set_axis_off()

    return fig


def multi_sized_images():
    random_id = randint(0, len(PBC_infos_df))
    img_path, cell_type = PBC_infos_df[["path", "cell_type"]].iloc[random_id]

    img = get_image(img_path)
    img_bg = img.convert('L')
    sizes = [None, 200, 100, 70, 50, 30]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Image d'une cellule de type {cell_type} dans différentes résolutions", fontsize=20)

    for i in range(2):
        for j in range(3):
            ax[i, j].set_axis_off()
            if i == 0 and j == 0:
                ax[0, 0].imshow(img)
                ax[i, j].set_title("Image originale")
            else:
                k = 3*i + j
                s = sizes[k]
                img_ = img_bg.resize((s, s))
                ax[i, j].imshow(img_, cmap="gray")
                ax[i, j].set_title(f"Image taille {s}x{s}")

    return fig


def plot_select_percentile_mask(size):

    paths = load_pickle_data("paths")
    target = load_pickle_data("target")

    selector = load_pickle_selector(f"select_percentile_{size}")
    mask = selector['mask']

    index = randint(0, len(paths))
    img = get_image(paths[index]).convert('L').resize((size, size))
    cell_type = target[index]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(img, cmap="gray")
    ax1.set_title(f"{cell_type} {size}x{size}")
    ax1.set_axis_off()

    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Masque")
    ax2.set_axis_off()

    ax3.imshow(np.where(mask, img, 0), cmap="gray")
    ax3.set_title(f"{cell_type} masqué(e)")
    ax3.set_axis_off()

    return fig


def reload_content(placeholder_func, func, *args):
    placeholder_func(func(*args))
