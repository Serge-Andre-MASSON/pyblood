import numpy as np
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from PIL import Image
from urllib.request import urlopen
import streamlit as st
from session.state import init_session_states, increment_counter
from data_viz.dl_plot import plot_predictions, CLASSES, predict_image

# TODO : Refactor all this, along with dl_plot
# TODO : Choose better urls


def get_demo(model_name, img_size=256):
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
            model_name, pred_counter, img_size)
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

        urls_by_cell_type = {'basophil': [
            "https://imagebank.hematology.org/getimagebyid/60504?size=3",
            "https://imagebank.hematology.org/getimagebyid/60505?size=3",
            "http://medcell.org/histology/blood_bone_marrow_lab/images/basophil.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Basophile-9.JPG/375px-Basophile-9.JPG",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Blausen_0077_Basophil_%28crop%29.png/375px-Blausen_0077_Basophil_%28crop%29.png"
        ],
            "eosinophil": [
                "http://imagebank.hematology.org/getimagebyid/60934?size=3",
                "http://imagebank.hematology.org/getimagebyid/60933?size=2",
                "http://www.hkimls.org/eos1.jpg",
                "http://www.hkimls.org/eos2.jpg"
        ],
            'erythroblast': [
                "https://www.cellavision.com/images/Hematopoiesis/Erythropoiesis/PolychromaticErythroblast/POLYCHROMATICERYTHROBLAST1.jpg",
                "https://www.cellavision.com/images/Hematopoiesis/Erythropoiesis/OrthochromaticErythroblast/ORTHROCHROMATICERYTHROBLAST1.jpg",
        ],
            'ig': ["A compléter"],
            'lymphocyte': ["A compléter"],
            'monocyte': ["A compléter"],
            'neutrophil': [
                "http://imagebank.hematology.org/getimagebyid/61935?size=2"
        ],
            'platelet': ["A compléter"]}
        # st.markdown("## Inférence sur des images tirées d'internet")
        cell_type = st.selectbox(
            "Type cellulaire", CLASSES)
        url = st.selectbox("Images", urls_by_cell_type[cell_type])

        # fig_1_, fig_2_, correctness, prediction = plot_predictions(
        #     model_name, pred_counter, img_size, url=url, cell_type=cell_type)
        # col_1_, col_2_ = st.columns(2)
        # with col_1_:
        #     st.pyplot(fig_1_)

        # with col_2_:
        #     st.pyplot(fig_2_)

        st.markdown(
            "## Inférence sur des images tirées d'internet")

        test_img = Image.open(urlopen(url))
        w, h = test_img.size

        test_img = test_img.resize((300, h * 300 // w))
        col1, col2 = st.columns(2)
        with col1:
            cropped_img = st_cropper(test_img, realtime_update=True, box_color='black',
                                     aspect_ratio=(1, 1))
        with col2:
            # st.image(cropped_img)
            fig, ax = plt.subplots()
            probas = predict_image(model_name, cropped_img)
            # pred_index = probas.argmax()
            ax.bar(CLASSES, height=predict_image(model_name, cropped_img))
            ax.tick_params(axis='x', labelrotation=45)
            st.pyplot(fig)
            # st.write(
            #     f"La prediction est {correctness} comme probable à {prediction.max()*100:.0f}%.")
    return demo


page_names_to_funcs = {
    "Dense net": get_demo('dense_net'),
    "Basic model": get_demo('basic_model', img_size=64),
    "vgg16": get_demo('vgg16'),
    "Inception V3": get_demo('inceptionv3'),
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
