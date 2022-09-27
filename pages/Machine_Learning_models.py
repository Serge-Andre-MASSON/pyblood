import streamlit as st


def section_1():
    st.markdown("## Présentation de la démarche")
    st.write("""Divers modèles de classification classiques ont été appliqués à ce problème pour évaluer leurs performances.
             On peut citer, entre autres, KNN, DecisionTree, SVC ou encore RandomForest.
             Afin d'améliorer les performances et réduire le temps de calcul,
             diverses étapes de pre-processing ont été appliquées aux images.
             D'abord de l'oversampling pour homognénéiser les effectifs des classes,
             puis de la réduction de dimensionalité à l'aide du rognage automatique présenté précédemment,
             suivi de SelectPercentile et enfin une PCA.
             Il s'avère que SVC et RandomForest sont les deux seuls modèles obtenant des performances satisfaisantes,
             et dans le cas de RandomForest aucun pre-processing n'est nécessaire, et les temps d'apprentissage est beaucoup plus faible.
             Afin d'évaluer l'influence de la taille des images sur les performances du modèle,
             les performances des divers modèles appliqués à chaque taille d'image (30 x 30, 50 x 50, 70 x 70, 100 x 100, 200 x 200) sont comparées.
             Le graphe ci-dessous présente les résultats obtenu pour le modèle SVC. 
             """)

    st.write("""On peut constater que les performances croissent avec le taille des images juqu'à 70 x 70,
             mais semblent ensuite osciller autour d'une valeur maximale.
             Cette taille constitue donc un bon compromis pour ces modèles,
             mais il est offert dans les sections suivantes de tester SVC et RandomForest, dont les hyperparamètres ont été optimisés,
             sur des images de toutes les tailles précédemment présentées.
             """)


def section_2():
    st.write(2)


page_names_to_funcs = {
    "Présentation de la démarche": section_1,
    "Section 2": section_2,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
