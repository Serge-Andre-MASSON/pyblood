import streamlit as st
from PIL import Image
from data_access.data_paths import get_figure_path
from data_access.data_access import get_image, get_random_image, load_ml_model
import matplotlib.pyplot as plt
from data_viz.plot import reload_content
from crop.crop import Crop
import numpy as np
from session.state import init_session_states, increment_counter


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

    image_path = get_figure_path(
        "data_images_performance_vs_size", extension='png')
    image = get_image(image_path)

    st.image(image)

    st.write("""On peut constater que les performances croissent avec le taille des images juqu'à 70 x 70,
             mais semblent ensuite osciller autour d'une valeur maximale.
             Cette taille constitue donc un bon compromis pour ces modèles,
             mais il est offert dans les sections suivantes de tester SVC et RandomForest, dont les hyperparamètres ont été optimisés,
             sur des images de toutes les tailles précédemment présentées.
             """)


def section_2():
    st.markdown("## Prédictions avec une Support Vector Machine")

    pixels_selection = st.selectbox("Mécanisme de sélection des pixels :", ['Cropping', 'Select Percentile (10%)'], index=0)

    size = st.selectbox("Taille des images en entrée du modèle :", [
                        100, 70, 50, 30], index=1)

    st.markdown("### Prédictions sur la base de données d'entraînement")

    model = load_ml_model('data/ml_models/svc_'+pixels_selection[0]+str(size)+'.joblib')

    pred_counter_key = f'prediction_counter_1'

    init_session_states(pred_counter_key)

    @st.experimental_memo
    def predict_image(counter):
        original_img, cell_type = get_random_image()
        img = original_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1, size**2)
        prediction = model.predict(img_data)
        pred_to_write = prediction[0]
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(cell_type)
        return fig, pred_to_write

    pred_counter = st.session_state[pred_counter_key]
    fig, pred_to_write = predict_image(pred_counter)
    st.write("Type cellulaire prédit :", pred_to_write)
    st.pyplot(fig)

    st.button(
        "Prédire une autre image",
        key=1,
        on_click=increment_counter,
        args=(pred_counter_key,))

    st.markdown("### Prédictions de vos images")

    user_file = st.file_uploader(label="Charger votre image")

    if user_file:
        user_img = Image.open(user_file)
        img = user_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1, size**2)
        prediction = model.predict(img_data)
        st.write("Type cellulaire prédit :", prediction[0])
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        fig_placeholder2 = st.empty()
        fig_placeholder2.pyplot(fig)


def section_3():

    st.markdown("## Prédictions avec une Random Forest")

    size = st.selectbox("Taille des images en entrée du modèle :", [
                        200, 100, 70, 50, 30], index=2)

    st.markdown("### Prédictions sur la base de données d'entraînement")

    model = load_ml_model('data/ml_models/rfc_'+str(size)+'.joblib')

    def predict_image():
        original_img, cell_type = get_random_image()
        img = original_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1, size**2)
        prediction = model.predict(img_data)
        st.write("Type cellulaire prédit :", prediction[0])
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
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

    user_file = st.file_uploader(label="Charger votre image")

    if user_file:
        user_img = Image.open(user_file)
        img = user_img.convert('L').resize((size, size))
        img_data = np.array(img).reshape(1, size**2)
        prediction = model.predict(img_data)
        st.write("Type cellulaire prédit :", prediction[0])
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        fig_placeholder2 = st.empty()
        fig_placeholder2.pyplot(fig)


page_names_to_funcs = {
    "Présentation de la démarche": section_1,
    "Support Vector Machine": section_2,
    "Random Forest": section_3
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
