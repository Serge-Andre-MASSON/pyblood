import streamlit as st
from session.state import init_session_states, increment_counter
from data_viz.dl_plot import (
    plot_mismatch_distribution,
    plot_pred_compare_with_truth,
    plot_predictions,
    CLASSES)


def get_demo(model_name):
    def demo():
        pred_counter_key = f'{model_name}_prediction_counter'

        init_session_states(pred_counter_key)

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
    return demo


page_names_to_funcs = {
    "Dense net": get_demo('dense_net'),
    "vgg16": get_demo('vgg16'),
    "Inception V3": get_demo('inceptionv3'),
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
