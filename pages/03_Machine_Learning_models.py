import streamlit as st
from PIL import Image
from data_access.data_paths import get_figure_path, get_ml_model_path
from data_access.data_access import get_random_image
from joblib import load
import matplotlib.pyplot as plt
from data_viz.plot import reload_content
from crop.crop import Crop
import numpy as np

def section_1():
    st.markdown("## Présentation de la démarche")
    st.write("""Divers modèles de classification classiques ont été appliqués à ce problème pour évaluer leurs performances.
             On peut citer, entre autres, KNN, DecisionTree, SVC ou encore RandomForest.
             Afin d'améliorer les performances et réduire le temps de calcul,
             diverses étapes de pre-processing ont été appliquées aux images.
             D'abord de l'oversampling pour homognénéiser les effectifs des classes,
             suivi de la réduction de dimensionalité à l'aide du rognage automatique présenté précédemment,
             et enfin une PCA.
             Il s'avère que SVC et RandomForest sont les deux seuls modèles obtenant des performances satisfaisantes,
             et dans le cas de RandomForest aucun pre-processing n'est nécessaire, et les temps d'apprentissage est beaucoup plus faible.
             Afin d'évaluer l'influence de la taille des images sur les performances du modèle,
             les performances des divers modèles appliqués à chaque taille d'image (30 x 30, 50 x 50, 70 x 70, 100 x 100, 200 x 200) sont comparées.
             Le graphe ci-dessous présente les résultats obtenu pour le modèle SVC. 
             """)


    image_path = get_figure_path("performance_vs_size")
    image = Image.open(image_path)

    st.image(image)

    st.write("""On peut constater que les performances croissent avec le taille des images juqu'à 70 x 70,
             mais semblent ensuite osciller autour d'une valeur maximale.
             Cette taille constitue donc un bon compromis pour ces modèles,
             mais il est offert dans les sections suivantes de tester SVC et RandomForest, dont les hyperparamètres ont été optimisés,
             sur des images de toutes les tailles précédemment présentées.
             """)


def section_2():
    st.markdown("## Prédictions avec une Support Vector Machine")
    
    size = st.selectbox("Taille des images en entrée du modèle :", [100, 70, 50, 30], index=1)
    
    st.markdown("### Prédictions sur la base de données d'entraînement")
    
    model_path = get_ml_model_path('svc_' + str(size))
    model = load(model_path)
    
    def predict_image():
        original_img, cell_type = get_random_image()
        img = original_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1,size**2)
        prediction = model.predict(img_data)
        st.write("Type cellulaire prédit :", prediction[0])
        fig = plt.figure()
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
        plt.title(cell_type)
        return fig
    
    fig = predict_image()
    fig_placeholder = st.empty()
    fig_placeholder.pyplot(fig)

    
    st.button(
        "Prédire une autre image",
        key=1,
        on_click=reload_content,
        args=(fig_placeholder.pyplot, predict_image))
    
    st.markdown("### Prédictions de vos images")
    
    user_file = st.file_uploader(label = "Charger votre image")
    
    if user_file:
        user_img = Image.open(user_file)
        img = user_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1,size**2)
        prediction = model.predict(img_data)
        st.write("Type cellulaire prédit :", prediction[0])
        fig = plt.figure()
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
        fig_placeholder2 = st.empty()
        fig_placeholder2.pyplot(fig)
        
def section_3():
    
    st.markdown("## Prédictions avec une Random Forest")
    
    size = st.selectbox("Taille des images en entrée du modèle :", [200, 100, 70, 50, 30], index=2)

    st.markdown("### Prédictions sur la base de données d'entraînement")
  
    model_path = get_ml_model_path('rfc_' + str(size))
    model = load(model_path)
    
    def predict_image():
        original_img, cell_type = get_random_image()
        img = original_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1,size**2)
        prediction = model.predict(img_data)
        st.write("Type cellulaire prédit :", prediction[0])
        fig = plt.figure()
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
        plt.title(cell_type)
        return fig
    
    fig = predict_image()
    fig_placeholder = st.empty()
    fig_placeholder.pyplot(fig)

    
    st.button(
        "Prédire une autre image",
        key=1,
        on_click=reload_content,
        args=(fig_placeholder.pyplot, predict_image))
    
    st.markdown("### Prédictions de vos images")
        
    user_file = st.file_uploader(label = "Charger votre image")
        
    if user_file:
        user_img = Image.open(user_file)
        img = user_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1,size**2)
        prediction = model.predict(img_data)
        st.write("Type cellulaire prédit :", prediction[0])
        fig = plt.figure()
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
        fig_placeholder2 = st.empty()
        fig_placeholder2.pyplot(fig)   

page_names_to_funcs = {
    "Présentation de la démarche": section_1,
    "Support Vector Machine": section_2,
    "Random Forest" : section_3
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
