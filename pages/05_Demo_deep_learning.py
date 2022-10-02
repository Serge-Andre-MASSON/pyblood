import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from data_access.data_access import get_image, load_pickle
from data_access.data_paths import get_dl_mismatch_path
from data_viz.dl_plot import plot_predictions, CLASSES


def init_session_state(*states):
    for state in states:
        if state not in st.session_state:
            st.session_state[state] = 0


def increment_counter(counter_state):
    st.session_state[counter_state] += 1


def describe_model(model_name):
    def images_from_original_dataset():
        pred_counter_key = f'{model_name}_session_counter'
        init_session_state(pred_counter_key)
        st.markdown(f"# {model_name}")
        st.markdown("## Inférence sur les images du dataset original")
        st.write("""Le modèle utilisé est le model dense net 121, initialisé avec les poids 'image_net',
        ajusté ensuite sur 80% du jeu de données. Les images choisies aléatoirement pour illustrer
        le fonctionnement du modèle proviennent indifférement du jeu d'entrainement ou du jeu de validation.""")

        col1, col2 = st.columns(2)
        pred_counter = st.session_state[pred_counter_key]
        fig_1, fig_2, correctness, prediction = plot_predictions(
            model_name, pred_counter)
        with col1:
            img_placeholder = st.empty()
            img_placeholder.pyplot(fig_1)

        with col2:
            barplot_placeholder = st.empty()
            barplot_placeholder.pyplot(fig_2)

            st.write(
                f"La prediction est {correctness} comme probable à {prediction.max()*100:.0f}%.")

        st.button("reload", on_click=increment_counter,
                  args=(pred_counter_key,))

        st.markdown("## Etude des erreurs faites par le modèle")
        st.markdown("### Répartition")

        mismatch_df_path = get_dl_mismatch_path(model_name)
        mismatch_df = load_pickle(mismatch_df_path)

        fig, ax = plt.subplots()

        sns.countplot(x=mismatch_df.true_cell_type,
                      hue=mismatch_df.predicted_cell_type, ax=ax)
        ax.tick_params(labelrotation=45)
        ax.set_title(
            "Répartition du type d'erreur en fonction du type cellulaire réel")
        ax.set_xlabel("Type cellulaire")
        ax.set_ylabel("type prédit")
        ax.set_yticks(range(0, 20, 4))
        ax.set_yticklabels([i for i in range(0, 20, 4)])
        ax.get_legend().set_title("Prédiction")

        st.pyplot(fig)

        st.markdown("### Visualisation")
        true_cell_type = st.selectbox("Type cellulaire réel:", CLASSES)
        cell_type_mimatch_df = mismatch_df[mismatch_df.true_cell_type ==
                                           true_cell_type]

        pred_cell_type = st.selectbox(
            "Type cellulaire prédit:", cell_type_mimatch_df.predicted_cell_type.unique())

        pred_cell_type_mimatch_df = cell_type_mimatch_df[
            cell_type_mimatch_df.predicted_cell_type == pred_cell_type].reset_index(drop=True)

        l = len(pred_cell_type_mimatch_df)
        st.write(
            f"Le type cellulaire {true_cell_type} est confondu {l} fois avec le type cellulaire {pred_cell_type}.")
        cell_by_row = 4
        fig, axes = plt.subplots(l // cell_by_row + 1,
                                 min(cell_by_row, l), figsize=(2*cell_by_row, l))
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]
        for i, ax in enumerate(axes):

            if i < l:
                path, _, pred = pred_cell_type_mimatch_df.iloc[i]
                img = get_image(path)
            else:
                img = np.zeros((256, 256, 3)) + 255
            ax.imshow(img)
            ax.set_axis_off()

        st.pyplot(fig)

        st.write(
            f"Ajouter ici quelques exemples du type cellulaire {pred_cell_type}.")
    return images_from_original_dataset


page_names_to_funcs = {
    "Dense net": describe_model('dense_net'),
    "vgg16": describe_model('vgg16'),
    "Inception V3": describe_model('inceptionv3'),
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
