import streamlit as st
from data_access.data_access import get_random_image, load_model
from data_access.data_paths import get_dl_model_path
import tensorflow as tf
from data_viz.plot import reload_content
import matplotlib.pyplot as plt

CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig',
           'lymphocyte', 'monocyte', 'neutrophil', 'platelet']


def images_from_original_dataset():
    st.markdown("# Inférence sur les images du dataset original")
    st.write("""Le modèle utilisé est le model dense net 121, initialisé avec les poids 'image_net',
    ajusté ensuite sur 80% du jeu de données. Les images choisies aléatoirement pour illustrer
    le fonctionnement du modèle proviennent indifférement du jeu d'entrainement ou du jeu de validation.""")

    def plot_predictions():
        img, target = get_random_image()
        img = img.resize((256, 256))
        tensor_img = tf.keras.utils.img_to_array(img).reshape(-1, 256, 256, 3)
        model_path = get_dl_model_path('dense_net')
        model = load_model(model_path)
        prediction = model.predict(tensor_img)[0]

        fig_1, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(target)
        ax.set_axis_off()

        fig_2, ax = plt.subplots()
        ax.bar(CLASSES, height=prediction)
        ax.set_title("Prévision")
        ax.tick_params(axis='x', labelrotation=45)
        correctness = "correcte et estimée" if target == CLASSES[
            prediction.argmax()] else "incorrecte, bien qu'estimée"
        return fig_1, fig_2, correctness, prediction

    col1, col2 = st.columns(2)
    fig_1, fig_2, correctness, prediction = plot_predictions()
    with col1:
        img_placeholder = st.empty()
        img_placeholder.pyplot(fig_1)

    with col2:
        barplot_placeholder = st.empty()
        barplot_placeholder.pyplot(fig_2)

        st.write(
            f"La prediction est {correctness} comme probable à {prediction.max()*100:.0f}%.")

    # Je n'ai pas encore trouvé la bonne méthode pour recharger un graphe...
    st.button("reload", on_click=plot_predictions)


page_names_to_funcs = {
    "Dense net": images_from_original_dataset,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
