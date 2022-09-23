from crop.crop import crop_np
from sklearn.base import TransformerMixin
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from data_access.data_access import (
    get_dataset_infos,
    get_image,
    load_pickle,
    get_random_image)
from crop.crop import crop_pil, get_limits, crop_np


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

    paths = load_pickle("data/PBC_pickles/paths.PICKLE")
    target = load_pickle("data/PBC_pickles/target.PICKLE")

    selector_path = f"data/feature_selection/select_percentile_{size}.PICKLE"
    selector = load_pickle(selector_path)
    mask = selector['mask']

    index = randint(0, len(paths) - 1)
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


def plot_gs_crop(size,):
    original_img, cell_type = get_random_image()
    img = original_img.resize((size, size)).convert('L')
    img_array = np.array(img)

    fig, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2, 2)

    ax_1.imshow(img, cmap='gray')
    ax_1.set_axis_off()
    ax_1.set_title(cell_type)

    i_up, i_down, to_i = get_limits(img_array, axis=1, size=size)
    j_left, j_right, to_j = get_limits(img_array, axis=0, size=size)

    std_i = img_array.std(axis=1)

    ax_2.plot(std_i, range(size))
    ax_2.hlines([i_up, i_down], 0, std_i.max(), colors='green')
    ax_2.invert_yaxis()
    ax_2.vlines(to_i, 0, size, colors='red')

    std_j = img_array.std(axis=0)

    ax_3.plot(std_j, label="Ecart type")
    ax_3.vlines([j_left, j_right], 0, std_j.max(), colors='green')
    ax_3.hlines(to_j, 0, size, colors='red', label='Seuil')

    ax_4.imshow(crop_np(img_array), cmap="gray")
    fig.legend()
    return fig


def plot_pca(size):
    data = load_pickle(f"data/PBC_pickles/data_{size}.PICKLE")
    targets = load_pickle(f"data/PBC_pickles/target.PICKLE")
    pca_summary = load_pickle(f"data/pca/pca_{size}.PICKLE")

    pca = pca_summary['pca']
    X = pca.transform(data)
    fig, ax = plt.subplots()
    for target in targets.unique():
        X_ = X[targets == target]
        # TODO: Add X_[:, 0], X_[:, 1] to the pickled pca to avoid inline computation
        ax.scatter(X_[:, 0], X_[:, 1], label=target)
    ax.set_title("Répartition des données selon les deux premières dimensions")
    ax.legend()
    return fig


def plot_sp_pca(size):
    data = load_pickle(f"data/PBC_pickles/data_{size}.PICKLE")
    targets = load_pickle(f"data/PBC_pickles/target.PICKLE")
    sp_pca_summary = load_pickle(f"data/pca/sp_pca_{size}.PICKLE")

    sp_pca = sp_pca_summary['pca']
    X = sp_pca.transform(data)

    fig, ax = plt.subplots()
    for target in targets.unique():
        X_ = X[targets == target]
        # TODO: Add X_[:, 0], X_[:, 1] to the pickled pca to avoid inline computation
        ax.scatter(X_[:, 0], X_[:, 1], label=target)
    ax.set_title("Répartition des données selon les deux premières dimensions")
    ax.legend()
    return fig


class Crop(TransformerMixin):
    img_size = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        self.img_size = np.sqrt(X.shape[-1]).astype('int')
        return self

    def crop(self, x):
        x_ = x.reshape((self.img_size, self.img_size))
        x_ = crop_np(x_, p=10)
        x_ = x_.reshape((self.img_size*self.img_size,))
        return x_

    def transform(self, X: np.ndarray):
        return np.apply_along_axis(self.crop, axis=1, arr=X)


def plot_crop_pca(size):
    data = load_pickle(f"data/PBC_pickles/data_{size}.PICKLE")
    targets = load_pickle(f"data/PBC_pickles/target.PICKLE")
    sp_pca_summary = load_pickle(f"data/pca/crop_pca_{size}.PICKLE")

    sp_pca = sp_pca_summary['pipeline']
    X = sp_pca.transform(data)

    fig, ax = plt.subplots()
    for target in targets.unique():
        X_ = X[targets == target]
        # TODO: Add X_[:, 0], X_[:, 1] to the pickled pca to avoid inline computation
        ax.scatter(X_[:, 0], X_[:, 1], label=target)
    ax.set_title("Répartition des données selon les deux premières dimensions")
    ax.legend()
    return fig


def reload_content(placeholder_func, func, *args):
    placeholder_func(func(*args))
