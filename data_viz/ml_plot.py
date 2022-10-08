# -*- coding: utf-8 -*-
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from data_access.data_paths import get_ml_mismatch_path
from data_access.data_access import load_pickle, get_image

CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig',
           'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

@st.experimental_memo
def plot_mismatch_distribution(model_name):
    mismatch_df_path = get_ml_mismatch_path(model_name)
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
def plot_pred_compare_with_truth(pred_cell_type_mimatch_df, size, cell_by_row=4):
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
            img = img.convert('L').resize((size, size))
        else:
            img = np.zeros((256, 256, 3)) + 255
        ax.imshow(img, cmap='gray')
        ax.set_axis_off()
    return fig