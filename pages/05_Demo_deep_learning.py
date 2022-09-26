import streamlit as st
from data_access.data_access import get_random_image
import tensorflow as tf
from data_viz.plot import reload_content
import matplotlib.pyplot as plt


@st.cache(allow_output_mutation=True)
def load_model(path):
    return tf.keras.models.load_model(path)


def images_from_original_dataset():
    st.markdown("# Inférence sur les images du dataset original")

    def plot_random_pred():
        img, target = get_random_image()
        img = img.resize((256, 256))
        tensor_img = tf.keras.utils.img_to_array(img).reshape(-1, 256, 256, 3)
        model = load_model("data/dl_models/dense_net_121_aft")
        prediction = model.predict(tensor_img)[0]

        fig, (ax_1, ax_2) = plt.subplots(2, 1)
        ax_1.imshow(img)
        ax_1.set_title(target)
        ax_1.set_axis_off()
        idx_to_class = {
            0: 'basophil',
            1: 'eosinophil',
            2: 'erythroblast',
            3: 'ig',
            4: 'lymphocyte',
            5: 'monocyte',
            6: 'neutrophil',
            7: 'platelet'
        }
        ax_2.bar([idx_to_class[i] for i in range(8)], height=prediction)
        ax_2.tick_params(axis='x', labelrotation=45)
        ax_2.set_title("Prévision")
        return fig

    preds_placeholder = st.empty()

    fig = plot_random_pred()
    preds_placeholder.pyplot(fig)
    st.button("reload", on_click=reload_content,
              args=(preds_placeholder.pyplot, plot_random_pred))


page_names_to_funcs = {
    "Section 1": images_from_original_dataset,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
