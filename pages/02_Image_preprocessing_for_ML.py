import streamlit as st
from data_viz.ml_img_preprocessing import (
    plot_bw_crop,
    plot_pca,
    plot_select_percentile_mask,
    plot_color_and_bw_img,
    multi_sized_images
)
from session.state import (
    init_session_states,
    increment_counter)


# Présentation du contenu sur la sidebar
st.sidebar.markdown("# Prétraitement des images")
st.sidebar.write("""On présente ici le travail fait sur le jeux de données initial
et sur les images pour optimiser la quantité et la qualité des données.""")


# Chaque section est contenue dans une fonction
def resampling():
    st.markdown("## Rééchantillonnage du jeux de données")
    st.write("à compléter")


def image_reduction():
    color_and_bw_counter_key = "color_and_bw_counter"
    multi_sized_images_counter_key = "multi_sized_images_counter"
    init_session_states(
        color_and_bw_counter_key,
        multi_sized_images_counter_key)

    st.markdown("## Réduction de la taille des données")

    st.markdown('### Images en niveaux de gris')
    st.write("""Etant donnée l'homogénéité du jeux de données en termes de
    palette de couleurs présente dans chaque image, on a estimé que se
    contenter de la version noir et blanc des images ne devrait pas représenter
    une grosse perte d'information.""")
    color_and_bw_counter = st.session_state[color_and_bw_counter_key]
    fig = plot_color_and_bw_img(color_and_bw_counter)
    st.pyplot(fig)

    st.button(
        "Charger une autre image",
        key=1,
        on_click=increment_counter,
        args=(color_and_bw_counter_key,))

    st.markdown('### Différentes tailles pour les images')

    st.write("""D'abord dans une volonté de pouvoir rapidement tester les algorithmes
    on a généré des jeux de données contenant les images en noir et blanc et en
    tailles réduites : de 30x30 à 200x200""")

    multi_sized_images_counter = st.session_state[
        multi_sized_images_counter_key]
    st.pyplot(multi_sized_images(multi_sized_images_counter))

    st.button(
        "Charger d'autres images",
        key=2,
        on_click=increment_counter,
        args=(multi_sized_images_counter_key,))

    st.write("""Lors des tests des différents algorithmes, on s'est rendu compte que
    réduire la taille de l'image avait un réel impact sur la qualité de la prédiction
    qu'au delà de 70x70.""")


def feature_selection():
    plot_sp_mask_counter_key = "plot_sp_mask_counter"
    plot_bw_crop_counter_key = "plot_gs_counter"
    init_session_states(
        plot_sp_mask_counter_key,
        plot_bw_crop_counter_key)
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
    plot_sp_mask_counter = st.session_state[plot_sp_mask_counter_key]
    fig = plot_select_percentile_mask(size, plot_sp_mask_counter)

    st.pyplot(fig)

    st.button(
        "Charger une autre image",
        key=3,
        on_click=increment_counter,
        args=(plot_sp_mask_counter_key,))

    st.markdown("""### Rognage automatique""")
    st.write("""On a mis au point un algorithme permettant dans la plupart des cas
    de repérer automatiquement la zone d'intérêt d'une image. L'idée de cet algorithme
    est de calculer l'écart type des valeurs des pixels selon les axes horizontaux
    et verticaux : lorsque l'écart type change sensiblement de valeur c'est que ce que
    l'on trouve le long de l'axe diffère sensiblement du fond de l'image.""")

    plot_bw_crop_counter = st.session_state[plot_bw_crop_counter_key]
    fig = plot_bw_crop(size, plot_bw_crop_counter)
    st.pyplot(fig)

    st.button("Recharger", on_click=increment_counter,
              args=(plot_bw_crop_counter_key,))


def dimension_reduction():
    st.markdown("## Réduction de dimension")
    st.write("On a réduit la dimension de notre jeu de données en utilisant l'algorithme principal Component Analysis (PCA).")

    size = st.selectbox("Taille des images :", [70, 50, 30], index=2)
    st.markdown("### PCA sur données brutes")
    st.write("Le résultat sur les données brutes est le suivant:")

    pca_fig, ratio = plot_pca(size)
    pca_placeholder = st.empty()
    pca_placeholder.pyplot(pca_fig)

    st.write(f"Pourcentage du jeu de données restant : {ratio}")

    st.markdown("### PCA après SelectPercentile")

    sp_pca_fig, ratio = plot_pca(size, selector='sp')
    sp_pca_placeholder = st.empty()
    sp_pca_placeholder.pyplot(sp_pca_fig)
    st.write(f"Pourcentage du jeu de données restant : {ratio}")

    st.markdown("### PCA après Rognage automatique")

    crop_pca_fig, ratio = plot_pca(size, selector='crop')
    crop_pca_placeholder = st.empty()
    crop_pca_placeholder.pyplot(crop_pca_fig)
    st.write(f"Pourcentage du jeu de données restant : {ratio}")


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
