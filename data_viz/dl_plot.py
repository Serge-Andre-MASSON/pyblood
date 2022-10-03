from data_access.data_access import get_image, get_random_image, load_dl_model, load_pickle
from data_access.data_paths import get_dl_mismatch_path, get_dl_model_path
import tensorflow as tf
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from urllib.request import urlopen


CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig',
           'lymphocyte', 'monocyte', 'neutrophil', 'platelet']


def load_model(model_name):
    model_path = get_dl_model_path(f'{model_name}')
    model = load_dl_model(model_path)
    return model


def predict_image(model_name, img: Image, img_size=256):
    img = img.resize((img_size, img_size))
    tensor_img = tf.keras.utils.img_to_array(
        img).reshape(-1, img_size, img_size, 3)
    model = load_model(model_name)
    prediction = model.predict(tensor_img)[0]
    return prediction


def predict_url(model, url: str, img_size_for_model=256):
    with Image.open(urlopen(url)) as i:
        img = i.copy()
    return predict_image(model, img)


@st.experimental_memo
def plot_predictions(model_name, counter, img_size=256, url="", cell_type=""):
    model = load_model(model_name)
    if not url:
        img, target = get_random_image()
    else:
        target = cell_type
        with Image.open(urlopen(url)) as i:
            img = i.copy()
    img = img.resize((img_size, img_size))
    tensor_img = tf.keras.utils.img_to_array(
        img).reshape(-1, img_size, img_size, 3)
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


@st.experimental_memo
def plot_mismatch_distribution(model_name):
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

    ax.set_yticks(range(0, len(mismatch_df)//5, len(mismatch_df)//12))
    ax.set_yticklabels(
        [i for i in range(0, len(mismatch_df)//5, len(mismatch_df)//12)])
    ax.get_legend().set_title("Prédiction")

    return fig, mismatch_df


@st.experimental_memo
def plot_pred_compare_with_truth(pred_cell_type_mimatch_df, cell_by_row=4):
    l = len(pred_cell_type_mimatch_df)
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
    return fig
