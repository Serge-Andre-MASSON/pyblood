import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from streamlit_cropper import st_cropper
import streamlit as st
from data_access.data_access import load_pickle
from data_access.data_paths import get_pbc_dataset_infos_paths

from session.state import increment_counter, init_session_states
from data_access.data_urls import urls_by_cell_type, get_image_by_url

from data_viz.dl_plot import (
    CLASSES,
    get_model_summary,
    plot_correct_pred,
    plot_mismatch_distribution,
    plot_pred_compare_with_truth,
    plot_prediction_for_a_dataset_random_image,
    plot_prediction_for_an_external_image)

# Présentation du contenu sur la sidebar
st.sidebar.markdown("# Présentation des CNN utilisés")
st.sidebar.write("""Dans cette partie, nous présentons les caractéristiques des divers réseaux de neurones testés ainsi que leurs performances.
Les résultats présentés sont ceux des transfer learning effectués à partir de chacun des modèles choisis. """)

COMMENTS = {
    "inceptionv3": """L’entraînement du modèle est effectué sur seulement 10 epochs
        et nous obtenons une accuracy de 92% contre 90% sans fine-tuning avec un
        temps d’entraînement d’environ 10 minutes. En revanche, nous remarquons
        un overfitting assez important avec ce modèle.""",

    "vgg16": """L’entraînement du modèle est effectué sur seulement 10 epochs mais
    se stoppe à 6 epochs car l’accuracy n’évolue plus et nous obtenons une accuracy
    de 95% contre 94% sans fine-tuning avec un temps d’entraînement d’environ 10
    minutes. En revanche, nous remarquons un léger overfitting avec ce modèle""",

    "dense_net": """L’entraînement du modèle est effectué sur seulement 25 epochs et
    nous obtenons une accuracy de 98% contre 95% sans fine-tuning avec un temps
    d’entraînement d’environ 50 minutes.""",

    "basic_model": """L’entraînement du modèle est effectué jusqu’à ce que l’accuracy
    de validation augmente de moins de 0.1% sur 10 epochs, avec également un mécanisme
    de réduction du taux d’apprentissage en cas de détection de plateau. Nous obtenons
    une accuracy de 96% avec un temps d’entraînement d’environ 30 minutes. Nous
    constatons une divergence entre accuracy d'entraînement et de validation à partir de la 20ème epoch environ.
    L’accuracy d’entraînement monte jusqu’aux alentours de 99% tandis que celle de
    validation plafonne autour de 96% à partir de la 25ème epoch environ."""
}


def get_section(model_name, img_size=256):
    def section():
        st.markdown(f"# Modèle de transfer learning avec {model_name}")
        st.markdown("## Paramètres du modèle et résultats")
        st.write(4*"\n")

        network_img, classification_report, accuracies = get_model_summary(
            model_name)
        # image du résumé du modèle
        st.image(network_img, width=800)
        st.write(COMMENTS[model_name])

        _, col2, _ = st.columns([0.2, 1, 0.2])
        # image du rapport de classification
        col2.image(classification_report, use_column_width=True, width=200)

        # image de la courbe accuracy
        col2.image(accuracies, use_column_width=True, width=200)

        st.markdown("## Etude des erreurs faites par le modèle")
        st.markdown("### Répartition")

        fig, mismatch_df = plot_mismatch_distribution(model_name)
        st.write(
            f"Il y a en tout {len(mismatch_df)} images mal classées pour ce modèle.")
        st.pyplot(fig)

        st.markdown("### Visualisation")
        true_cell_type = st.selectbox("Type cellulaire réel:", CLASSES)
        cell_type_mismatch_df = mismatch_df[mismatch_df.true_cell_type ==
                                            true_cell_type]
        st.markdown(f"##### Exemple de {true_cell_type}s correctement prédits.")
        true_correct_pred_counter_key = f"{model_name}_true_correct_pred_counter_key"
        init_session_states(true_correct_pred_counter_key)

        correct_pred_counter = st.session_state[true_correct_pred_counter_key]
        fig = plot_correct_pred(
            true_cell_type, cell_type_mismatch_df, correct_pred_counter)

        st.pyplot(fig)
        st.button("Voir d'autres", key=1, on_click=increment_counter,
                  args=(true_correct_pred_counter_key,))

        pred_cell_type = st.selectbox(
            "Type cellulaire prédit:", cell_type_mismatch_df.predicted_cell_type.unique())

        pred_cell_type_mismatch_df = cell_type_mismatch_df[
            cell_type_mismatch_df.predicted_cell_type == pred_cell_type].reset_index(drop=True)

        l = len(pred_cell_type_mismatch_df)
        st.write(
            f"Le type cellulaire {true_cell_type} est confondu {l} fois avec le type cellulaire {pred_cell_type}.")

        fig = plot_pred_compare_with_truth(pred_cell_type_mismatch_df)
        st.pyplot(fig)

        st.markdown(
            f"##### Exemple de {pred_cell_type}s correctement prédits.")
        pred_correct_pred_counter_key = f"{model_name}_pred_correct_pred_counter_key"

        init_session_states(pred_correct_pred_counter_key)

        pred_correct_pred_counter = st.session_state[pred_correct_pred_counter_key]
        fig = plot_correct_pred(
            pred_cell_type, pred_cell_type_mismatch_df, pred_correct_pred_counter)

        st.pyplot(fig)
        st.button("Voir d'autres", key=2, on_click=increment_counter,
                  args=(pred_correct_pred_counter_key,))

        st.markdown("## Inférence sur images du jeu de données original")
        pred_counter_key = f'{model_name}_prediction_counter'

        init_session_states(pred_counter_key)

        st.markdown("### Jeux de données original")
        col_1_1, col_1_2 = st.columns(2)
        pred_counter = st.session_state[pred_counter_key]

        fig_1_1, fig_1_2, pred_eval = plot_prediction_for_a_dataset_random_image(
            model_name, pred_counter, img_size)

        with col_1_1:
            st.pyplot(fig_1_1)
        with col_1_2:
            st.pyplot(fig_1_2)
            st.write(pred_eval)

        st.button("Recharger", on_click=increment_counter,
                  args=(pred_counter_key,))

        st.markdown(
            "## Images externes au jeu de données original")

        cell_type = st.selectbox(
            "Type cellulaire", CLASSES)
        url = st.selectbox("Images", urls_by_cell_type[cell_type])

        url_img = get_image_by_url(url)

        # Les lignes ci-dessous permettent de diminiuer la taille réelle de l'image
        # pour que cette dernière tienne dans la colonne : elle ne s'adapte pas
        # automatiquement au contenant.
        w, h = url_img.size
        max_width = 350
        url_img = url_img.resize((max_width, h * max_width // w))

        col1, col2 = st.columns(2)
        with col1:
            cropped_img = st_cropper(url_img, realtime_update=True, box_color='black',
                                     aspect_ratio=(1, 1))
        with col2:
            fig, pred_eval = plot_prediction_for_an_external_image(
                model_name, cropped_img, cell_type, img_size)
            st.pyplot(fig)
            st.write(pred_eval)

    return section


page_names_to_funcs = {
    "Basic Model": get_section("basic_model", img_size=64),
    "DenseNet121": get_section("dense_net"),
    "VGG16": get_section("vgg16"),
    "InceptionV3": get_section("inceptionv3"),
}

selected_page = st.sidebar.selectbox(
    "Section : ", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
