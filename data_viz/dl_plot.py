from data_access.data_access import get_random_image, load_dl_model
from data_access.data_paths import get_dl_model_path
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt


CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig',
           'lymphocyte', 'monocyte', 'neutrophil', 'platelet']


@st.experimental_memo
def plot_predictions(model_name, counter):
    img, target = get_random_image()
    img = img.resize((256, 256))
    tensor_img = tf.keras.utils.img_to_array(img).reshape(-1, 256, 256, 3)
    model_path = get_dl_model_path(f'{model_name}')
    model = load_dl_model(model_path)
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
