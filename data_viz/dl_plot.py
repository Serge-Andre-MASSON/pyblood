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


def load_model(model_name):
    model_path = get_dl_model_path(f'{model_name}')
    model = load_dl_model(model_path)
    return model


def predict_image(model, img: Image, img_size=256):
    img = img.resize((img_size, img_size))
    tensor_img = tf.keras.utils.img_to_array(
        img).reshape(-1, img_size, img_size, 3)
    prediction = model.predict(tensor_img)[0]
    return prediction


def get_pred_evaluation(cell_type, prediction):
    predicted_cell_type = prediction.argmax()
    correctness = "correcte et estimée" if cell_type == CLASSES[
        predicted_cell_type] else "incorrecte, bien qu'estimée"
    return f"La prediction est {correctness} comme probable à {prediction.max()*100:.0f}%."


def plot_prediction(prediction):
    fig, ax = plt.subplots()
    ax.bar(CLASSES, height=prediction)
    ax.set_title("Prévision")
    ax.tick_params(axis='x', labelrotation=45)
    return fig


@st.experimental_memo
def plot_prediction_for_a_dataset_random_image(model_name, counter, img_size=256):
    model = load_model(model_name)
    img, cell_type = get_random_image()
    prediction = predict_image(model, img, img_size=img_size)

    fig_1, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(cell_type)
    ax.set_axis_off()

    fig_2 = plot_prediction(prediction)

    return fig_1, fig_2, get_pred_evaluation(cell_type, prediction)


def plot_prediction_for_an_external_image(
        model_name, img, cell_type, img_size=256):

    model = load_model(model_name)
    prediction = predict_image(model, img, img_size=img_size)

    fig = plot_prediction(prediction)
    pred_eval = get_pred_evaluation(cell_type, prediction)
    return fig, pred_eval
