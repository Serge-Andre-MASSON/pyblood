from random import randint

import streamlit as st
import matplotlib.pyplot as plt
from data_access import get_dataset_infos, get_image

st.title('PBC dataset')
st.write('Le dataframe ci-dessous contient quelques informations à propos de la base de données sur laquelle on travaille.')

PBC_infos_df = get_dataset_infos()
st.write(PBC_infos_df)
st.subheader('Visualisation des images')


def plot_cell_types():
    cell_types = PBC_infos_df.cell_type.unique()
    fig, axes = plt.subplots(2, 4)

    for cell_type, ax in zip(cell_types, axes.flatten()):

        cell_type_df = PBC_infos_df[PBC_infos_df.cell_type ==
                                    cell_type].reset_index()
        n = len(cell_type_df)
        id_ = randint(0, n-1)

        img_path = cell_type_df.path[id_]
        img = get_image(img_path)
        ax.imshow(img)
        ax.set_title(cell_type)
        ax.axis('off')

    return fig


placeholder = st.empty()
placeholder.pyplot(plot_cell_types())

reload_ = st.button("Charger d'autres images")
if reload_:
    with placeholder.container():
        st.pyplot(plot_cell_types())
    reload_ = False

st.subheader('Distribution des types cellulaires')


cell_types = PBC_infos_df['cell_type'].value_counts(sort=False)
fig, ax = plt.subplots(1, 1)
ax.bar(cell_types.index, height=cell_types.values)
ax.tick_params(axis='x', labelrotation=45)
st.pyplot(fig)
