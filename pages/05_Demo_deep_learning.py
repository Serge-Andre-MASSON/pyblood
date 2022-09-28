import streamlit as st
from data_viz.dl_plot import plot_predictions


def init_session_state(*states):
    for state in states:
        if state not in st.session_state:
            st.session_state[state] = 0


def increment_counter(counter_state):
    st.session_state[counter_state] += 1


def images_from_original_dataset():

    init_session_state('pred_counter')
    st.markdown("# Dense Net 121")
    st.markdown("## Inférence sur les images du dataset original")
    st.write("""Le modèle utilisé est le model dense net 121, initialisé avec les poids 'image_net',
    ajusté ensuite sur 80% du jeu de données. Les images choisies aléatoirement pour illustrer
    le fonctionnement du modèle proviennent indifférement du jeu d'entrainement ou du jeu de validation.""")

    col1, col2 = st.columns(2)
    pred_counter = st.session_state['pred_counter']
    fig_1, fig_2, correctness, prediction = plot_predictions(pred_counter)
    with col1:
        img_placeholder = st.empty()
        img_placeholder.pyplot(fig_1)

    with col2:
        barplot_placeholder = st.empty()
        barplot_placeholder.pyplot(fig_2)

        st.write(
            f"La prediction est {correctness} comme probable à {prediction.max()*100:.0f}%.")

    st.button("reload", on_click=increment_counter, args=('pred_counter',))

    st.markdown("## Visualisation des erreurs")


page_names_to_funcs = {
    "Dense net": images_from_original_dataset,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
