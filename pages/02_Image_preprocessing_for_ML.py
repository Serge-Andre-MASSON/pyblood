from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from data_access.data_access import get_random_image
from data_viz.plot import plot_select_percentile_mask, reload_content, plot_color_and_bg_img, multi_sized_images


# Présentation du contenu sur la sidebar
st.sidebar.markdown("# Prétraitement des images")
st.sidebar.write("""On présente ici le travail fait sur le jeux de données initial
et sur les images pour optimiser la quantité et la qualité des données.""")


# Chaque section est contenue dans une fonction
def resampling():
    st.markdown("## Rééchantillonnage du jeux de données")
    st.write("à compléter")


def image_reduction():
    st.markdown("## Réduction de la taille des données")

    st.markdown('### Images en niveaux de gris')
    st.write("""Etant donnée l'homogénéité du jeux de données en termes de
    palette de couleurs présente dans chaque image, on a estimé que se
    contenter de la version noir et blanc des images ne devrait pas représenter
    une grosse perte d'information.""")

    fig_placeholder = st.empty()
    fig = plot_color_and_bg_img()
    fig_placeholder.pyplot(fig)

    st.button(
        "Charger une autre image",
        key=1,
        on_click=reload_content,
        args=(fig_placeholder.pyplot, plot_color_and_bg_img))
    st.markdown('### Différentes tailles pour les images')

    st.write("""D'abord dans une volonté de pouvoir rapidement tester les algorithmes
    on a généré des jeux de données contenant les images en noir et blanc et en
    tailles réduites : de 30x30 à 200x200""")

    placeholder = st.empty()
    placeholder.pyplot(multi_sized_images())

    st.button(
        "Charger d'autres images",
        key=2,
        on_click=reload_content,
        args=(placeholder.pyplot, multi_sized_images))

    st.write("""Lors des tests des différents algorithmes, on s'est rendu compte que
    réduire la taille de l'image avait un réel impact sur la qualité de la prédiction
    qu'au delà de 70x70.""")


def feature_selection():
    st.markdown("## Selection de features")

    st.markdown("### SelectPercentile")
    st.write("""Dans un premier temps on a utilisé l'algorithme SelectPercentile de la
    bibliothèque scikit-learn.""")
    st.write("""Ce dernier effectue un test ANOVA entre les pixels de l'image et leur label
    puis retourne un tableau contenant le score de chaque pixel si celui-ci
    est dans le top 10 pour cent (valeur par défaut)  et 0 sinon.
    On peut alors construire un masque permettant de retenir les pixels
    obtenant les meilleurs scores.""")

    size = st.selectbox("Taille des images :", [200, 100, 70, 50, 30], index=2)

    placeholder_0 = st.empty()
    fig = plot_select_percentile_mask(size)

    placeholder_0.pyplot(fig)

    st.button(
        "Charger une autre image",
        key=3,
        on_click=reload_content,
        args=(placeholder_0.pyplot, plot_select_percentile_mask, size))

    st.markdown("""### Rognage automatique""")
    st.write("""On a mis au point un algorithme permettant dans la plupart des cas
    de repérer automatiquement la zone d'intérêt d'une image. L'idée de cet algorithme
    est de calculer l'écart type des valeurs des pixels selon les axes horizontaux
    et verticaux : lorsque l'écart type change sensiblement de valeur c'est que ce que 
    l'on trouve le long de l'axe diffère sensiblement du fond de l'image.""")

    def get_limits(img, axis, size):
        std_img = img.std(axis=axis)
        t = threshold_otsu(std_img)

        t_std_img = std_img > t * 1.2

        j_min = np.argmax(t_std_img)
        j_max = size - np.argmax(t_std_img[::-1])
        return j_min, j_max, t*1.2

    def crop(img_: np.ndarray, p=10):
        img = img_.copy()
        height, width = img.shape

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

    def plot_crop(size):
        original_img, cell_type = get_random_image()
        img = original_img.convert('L').resize((size, size))

        fig, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2, 2)
        img_array = np.array(img)

        ax_1.imshow(img_array, cmap="gray")
        ax_1.set_axis_off()
        ax_1.set_title(cell_type)

        i_up, i_down, to_i = get_limits(img_array, axis=1, size=size)
        j_left, j_right, to_j = get_limits(img_array, axis=0, size=size)

        std_i = img_array.std(axis=1)

        ax_2.plot(std_i, range(size))
        ax_2.hlines([i_up, i_down], 0, std_i.max(), colors='green')
        ax_2.invert_yaxis()
        ax_2.vlines(to_i, 0, size, colors='red', label='threshold')

        std_j = img_array.std(axis=0)

        ax_3.plot(std_j)
        ax_3.vlines([j_left, j_right], 0, std_j.max(), colors='green')
        ax_3.hlines(to_j, 0, size, colors='red', label='threshold')

        ax_4.imshow(crop(img_array, 15), cmap="gray")
        return fig

    placeholder_1 = st.empty()
    fig = plot_select_percentile_mask(size)

    fig = plot_crop(size)
    placeholder_1.pyplot(fig)

    st.button("Recharger", on_click=reload_content,
              args=(placeholder_1.pyplot, plot_crop, size))


def dimension_reduction():
    st.markdown("## Réduction de dimension")
    st.markdown("### PCA")
    st.write("à compléter")
    st.markdown("### LDA")
    st.write("à compléter")


# Ce bloc de code permet de passer d'une section à une autre via la dropbox de la sidebar
page_names_to_funcs = {
    "Rééchantillonage du jeux de données": resampling,
    "Réduction de la taille des images": image_reduction,
    "Selection de features": feature_selection,
    "réduction de dimension": dimension_reduction
}

selected_page = st.sidebar.selectbox(
    "Section : ", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
