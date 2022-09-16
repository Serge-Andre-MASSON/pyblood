from random import randint

import streamlit as st
import matplotlib.pyplot as plt
from data_access.data_access import get_dataset_infos, get_image
from data_viz.plot import all_cell_types, cell_types_distribution, reload_content


st.sidebar.markdown("# Pyblood")
st.sidebar.write("""On donne ici un aperçu rapide des objectifs du projet et 
du contenu du jeu de données sur lequel on va travailler.""")


st.title('Pyblood')
st.subheader('Objectif')

st.write("""L'objectif de ce projet est de proposer un modèle de classification
d'images en fonction du type cellulaire présent sur ces dernières.""")
st.write("""Le modèle de classification se base sur le jeu de données 
PBC_dataset_normal_DIB, contenant 17092 images de microscopie 
d’individus sains anonymisés""")
st.write("""Les cellules à identifier sont classées selon 8 types cellulaires: 
monocytes, lymphocytes, plaquettes (platelet), ig, basophiles, éosinophiles, granulocytes(neutrophil) et érythroblastes.""")

st.subheader("Visualisation des données")
st.write("""Les images à classer selon le type cellulaire se présentent ainsi : """)

PBC_infos_df = get_dataset_infos()

placeholder = st.empty()
placeholder.pyplot(all_cell_types())


st.button("Charger d'autres images",
          on_click=reload_content, args=(placeholder.pyplot, all_cell_types))

st.write("""On note que dans la plupart des cas, l'information essentielle 
se situe au centre de l'image. On proposera deux façons de tirer parti de ce constat : """)

st.markdown("""- Un algorithme de selection des pixels se basant sur 
l'impact de ces dernier dans la variance du type cellulaire""")
st.markdown("- Un algorithme de rognage automatique de l'image")
st.subheader("Distribution des données")
st.write("""La distribution des types cellulaires au sein du jeu de données se résume ainsi :""")


target = PBC_infos_df.cell_type
fig = cell_types_distribution(target)
st.pyplot(fig)

st.write("""La distribution des données étant déséquilibrée, on utilisera des algorithmes de reéchantillonnage 
dans le but d'améliorer les prédictions.""")
