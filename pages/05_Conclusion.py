import streamlit as st
from data_access.data_access import get_image
from data_access.data_paths import get_figure_path

st.title('Conclusion & Perspectives')
st.subheader('Conception du projet')

st.write("Nous avons procédé par étapes:")
st.markdown(
    """- Dataviz: visualisation des données sous forme graphiques et en affichant les images.""")
st.markdown(
    """- Preprocessing des données: outils de sélection et de réduction de features sur les images.""")
st.markdown("""- Machine Learning: développement de plusieurs modèles simples de classification puis plus complexes avec du Bagging et du Boosting.""")
st.markdown("""- Deep Learning: développement de divers réseaux de neurones simples puis application de méthodes de transfert learning.""")

img1 = get_image(get_figure_path("chronologie"))
st.image(img1, width=800)

st.subheader('Outils utilisés')
st.write("Nous avons considéré ce projet comme une manière d’explorer tout le savoir-faire acquis pendant la formation et avons valorisé l'utilisation de plusieurs d’outils.")

img2 = get_image(get_figure_path("outils"))
st.image(img2, width=800)

st.subheader("Bilan des observations")
st.write("Nous pouvons conclure que la solution proposée pour la classification des images répond bien à la problématique de ce projet. Le modèle retenu est capable de classer les images avec un taux de bonnes prédictions avoisinant les 98% ce qui est une performance plus que correcte.")
st.write("\n")
st.write("Nous avons pu identifier les sources potentielles d'erreurs faîtes par notre modèle:")
st.markdown("""- Erreur de labellisation par le médecin.""")
st.markdown("""- Apparition de 2 cellules sur image au lieu d'une seule.""")
st.markdown("""- La complexité de la biologie humaine.""")

st.subheader("Perspectives")
st.write("Il serait intéressant de poursuivre notre étude afin de proposer un modèle de classification de cellules anormales. De faire identifier à notre modèle les particularités qui lui font classer la cellule comme anormale. Et enfin, de définir une pathologie potentielle associée à cette anormalité pour aider le praticien à poser son diagnostic final.")
st.write("Nous pourrions faire de la segmentation afin de compter les cellules de chaque type cellulaire dans un frottis.")

st.subheader("Interprétabilité")
st.write("Les modèles proposés font l’objet de zones d’ombres dans leur prise de décision (boîtes noires), qui rendent parfois difficile d’accès l’interprétabilité de ces modèles pour le corps médical. Il est donc nécessaire d'améliorer leur compréhension")
st.write("Une première approche que nous avons abordé dans ce projet pour améliorer l'interprétabilité de notre est l'affichage des features sélectionnées par notre réseau de neurones.")
