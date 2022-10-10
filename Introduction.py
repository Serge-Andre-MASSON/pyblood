import streamlit as st
from data_access.data_access import load_pickle
from data_access.data_paths import get_pbc_dataset_infos_paths
from data_viz.plot import all_cell_types, cell_types_distribution, reload_content
from data_access.data_access import get_image
from data_access.data_paths import get_figure_path


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

targets_path = get_pbc_dataset_infos_paths('targets')
targets = load_pickle(targets_path)

placeholder = st.empty()
placeholder.pyplot(all_cell_types(targets))


st.button("Charger d'autres images",
          on_click=reload_content, args=(placeholder.pyplot, all_cell_types, targets))

st.write("""On note que dans la plupart des cas, l'information essentielle 
se situe au centre de l'image. On proposera deux façons de tirer parti de ce constat : """)

st.markdown("""- Un algorithme de selection des pixels se basant sur l'impact de ces dernier dans la variance du type cellulaire""")
st.markdown("- Un algorithme de rognage automatique de l'image")

st.title('Identification de biais')
st.subheader('Notions de bases en biologie')

img1 = get_image(get_figure_path("hematopoiese"))
st.image(img1, width=800)

st.write("""Différencier les types cellulaires de ce projet en 8 classes demande à comprendre quelques notions de base en biologie.
En effet, la différenciation cellulaire est une science complexe et certains biais peuvent venir perturber nos modèles
de prédiction.""")

st.subheader("Cas concrets")

img2 = get_image(get_figure_path("differenciation1"))
st.image(img2, width=800)
img3 = get_image(get_figure_path("differenciation2"))
st.image(img3, width=800)

st.write("""Ceci n'explique pas complètement les erreurs que nos modèles feront par la suite, cependant cela met
en évidence la difficulté de ce projet et la question scientifique qu'il faut résoudre à la fin.""")
