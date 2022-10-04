from random import randint

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data_access.data_access import get_image, get_random_image, load_pickle
from data_access.data_paths import get_pbc_dataset_infos_paths, get_transformer_path
from crop.crop import get_limits, crop_np


PBC_infos_df = load_pickle(
    get_pbc_dataset_infos_paths('both')
)


@st.experimental_memo
def plot_color_and_bw_img(counter):

    random_id = randint(0, len(PBC_infos_df) - 1)
    img_path, cell_type = PBC_infos_df.iloc[random_id]

    img = get_image(img_path)
    img_bg = img.convert('L')

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(cell_type)
    axes[0].imshow(img)
    axes[0].set_axis_off()
    axes[1].imshow(img_bg, cmap='gray')
    axes[1].set_axis_off()

    return fig


@st.experimental_memo
def multi_sized_images(counter):
    random_id = randint(0, len(PBC_infos_df))
    img_path, cell_type = PBC_infos_df.iloc[random_id]

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


@st.experimental_memo
def plot_select_percentile_mask(size, counter):

    selector_path = get_transformer_path(size, 'sp')
    selector = load_pickle(selector_path)
    mask = selector['mask']

    img, cell_type = get_random_image()
    img = img.convert('L').resize((size, size))
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


@st.experimental_memo
def plot_bw_crop(size, counter):
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


def plot_pca(size, selector=None):
    if selector:
        path = get_transformer_path(size, selector, 'pca')
    else:
        path = get_transformer_path(size, 'pca')

    pca = load_pickle(path)
    targets = pca['targets']

    X_0 = pca['first_dimension']
    X_1 = pca['second_dimension']

    fig, ax = plt.subplots()
    for target in targets.unique():
        X_0_ = X_0[targets == target]
        X_1_ = X_1[targets == target]
        ax.scatter(X_0_, X_1_, label=target)
    ax.set_title("Répartition des données selon les deux premières dimensions")
    ax.legend()
    ratio = pca['feature_size_after'] / pca["feature_size_before"]
    return fig, ratio
