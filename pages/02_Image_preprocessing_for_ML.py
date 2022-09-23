from data_access.data_access import load_pickle
import streamlit as st
from data_viz.plot import (
    plot_crop_pca,
    plot_gs_crop,
    plot_pca,
    plot_select_percentile_mask,
    plot_sp_pca,
    reload_content,
    plot_color_and_bg_img,
    multi_sized_images
)


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

    placeholder_1 = st.empty()

    fig = plot_gs_crop(size)
    placeholder_1.pyplot(fig)

    st.button("Recharger", on_click=reload_content,
              args=(placeholder_1.pyplot, plot_gs_crop, size))


def dimension_reduction():
    st.markdown("## Réduction de dimension")
    st.write("On a réduit la dimension de notre jeu de données en utilisant l'algorithme principal Component Analysis (PCA).")
    st.markdown("### PCA sur données brutes")
    st.write("Le résultat sur les données brutes est le suivant:")

    size = st.selectbox("Taille des images :", [100, 70, 50, 30], index=3)

    pca_fig = plot_pca(size)
    pca_placeholder = st.empty()
    pca_placeholder.pyplot(pca_fig)

    st.markdown("### PCA après SelectPercentile")

    sp_pca_fig = plot_sp_pca(size)
    sp_pca_placeholder = st.empty()
    sp_pca_placeholder.pyplot(sp_pca_fig)
    st.write(":")

    st.markdown("### PCA après Rognage automatique")

    st.write("A implémenter")
    # TODO: réparer les pickles

    # crop_pca_fig = plot_crop_pca(size)
    # crop_pca_placeholder = st.empty()
    # crop_pca_placeholder.pyplot(crop_pca_fig)
    # st.write(":")


###########################################################################################
# Ce bloc de code permet de passer d'une section à une autre via la dropbox de la sidebar #
###########################################################################################
page_names_to_funcs = {
    "Rééchantillonage du jeux de données": resampling,
    "Réduction de la taille des images": image_reduction,
    "Selection de features": feature_selection,
    "réduction de dimension": dimension_reduction
}

selected_page = st.sidebar.selectbox(
    "Section : ", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
