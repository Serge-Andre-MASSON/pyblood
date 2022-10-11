import streamlit as st
from PIL import Image
from data_access.data_paths import get_figure_path
from data_access.data_access import get_image, get_random_image, load_ml_model
import matplotlib.pyplot as plt
from crop.crop import Crop
import numpy as np
from session.state import init_session_states, increment_counter
from data_viz.ml_plot import plot_correct_pred, plot_mismatch_distribution, plot_pred_compare_with_truth, CLASSES
from data_access.data_urls import urls_by_cell_type, get_image_by_url
from streamlit_cropper import st_cropper

# Présentation du contenu sur la sidebar
st.sidebar.markdown("# Modèles de Machine Learning")
st.sidebar.write(
    """On présente ici les résultats obtenus dans la tâche d'identification de type cellulaire à l'aide de modèles de Machine Learning classiques.""")


def section_1():
    st.markdown("# Présentation de la démarche")
    st.write("""Divers modèles de classification classiques ont été appliqués à ce problème pour évaluer leurs performances.
             On peut citer, entre autres, KNN, DecisionTree, SVC ou encore RandomForest.""")

    st.write("""Afin d'améliorer les performances et réduire le temps de calcul,
             diverses étapes de pre-processing ont été appliquées aux images converties en niveaux de gris.
             D'abord de l'oversampling pour homognénéiser les effectifs des classes,
             suivi d'une sélection de features à l'aide du rognage automatique ou de SelectPercentile, présentés précédemment,
             et enfin une réduction de dimensionalité à l'aide de la PCA.""")

    st.write("""Il s'avère que SVC et RandomForest sont les deux seuls modèles obtenant des performances satisfaisantes,
             et dans le cas de RandomForest aucun pre-processing n'est nécessaire, et les temps d'apprentissage sont beaucoup plus faibles.""")


def section_2():
    st.markdown("# Prédictions avec une Support Vector Machine")

    pixels_selection = st.selectbox("Mécanisme de sélection des pixels :", [
                                    'Cropping', 'Select Percentile (10%)'], index=0)

    if pixels_selection == 'Select Percentile (10%)':
        size = st.selectbox("Taille des images en entrée du modèle :", [
                            200, 100, 70, 50, 30], index=2)
    else:
        size = st.selectbox("Taille des images en entrée du modèle :", [
                            100, 70, 50, 30], index=1)

    model_name = 'svc_'+pixels_selection[0]+str(size)

    st.markdown("## Performances générales sur la base d'apprentissage")

    image_path = get_figure_path(
        'rapport_classification_'+model_name, extension='png')
    image = get_image(image_path)

    st.image(image)

    st.markdown("## Prédictions sur la base d'apprentissage")

    model = load_ml_model('data/ml_models/'+model_name+'.joblib')

    pred_counter_key1 = f'prediction_counter_1'

    init_session_states(pred_counter_key1)

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

    pred_counter = st.session_state[pred_counter_key1]
    fig, pred_to_write = predict_image(pred_counter)
    st.write("Type cellulaire prédit :", pred_to_write)
    st.pyplot(fig)

    st.button(
        "Prédire une autre image",
        key=1,
        on_click=increment_counter,
        args=(pred_counter_key1,))

    st.markdown("## Etude des erreurs faites par le modèle")
    st.markdown("### Répartition")

    fig, mismatch_df = plot_mismatch_distribution(model_name)
    st.write(
        f"Il y a en tout {len(mismatch_df)} images mal classées pour ce modèle.")
    st.pyplot(fig)

    st.markdown("### Visualisation")
    true_cell_type = st.selectbox("Type cellulaire réel:", CLASSES)
    cell_type_mismatch_df = mismatch_df[mismatch_df.true_cell_type ==
                                        true_cell_type]

    st.markdown(f"##### Exemple de {true_cell_type}s correctement prédits.")

    correct_pred_counter_key = f"{model_name}_correct_pred_counter_key"
    init_session_states(correct_pred_counter_key)

    correct_pred_counter = st.session_state[correct_pred_counter_key]
    fig = plot_correct_pred(
        true_cell_type, cell_type_mismatch_df, correct_pred_counter, size)

    st.pyplot(fig)
    st.button("Voir d'autres", on_click=increment_counter,
              args=(correct_pred_counter_key,))

    pred_cell_type = st.selectbox(
        "Type cellulaire prédit:", cell_type_mismatch_df.predicted_cell_type.unique())

    pred_cell_type_mismatch_df = cell_type_mismatch_df[
        cell_type_mismatch_df.predicted_cell_type == pred_cell_type].reset_index(drop=True)

    l = len(pred_cell_type_mismatch_df)
    st.write(
        f"Le type cellulaire {true_cell_type} est confondu {l} fois avec le type cellulaire {pred_cell_type}.")

    fig = plot_pred_compare_with_truth(pred_cell_type_mismatch_df, size=size)
    st.pyplot(fig)
    
    st.markdown(
            f"##### Exemple de {pred_cell_type} correctement prédits.")
    pred_correct_pred_counter_key = f"{model_name}_pred_correct_pred_counter_key"

    init_session_states(pred_correct_pred_counter_key)

    pred_correct_pred_counter = st.session_state[pred_correct_pred_counter_key]
    fig = plot_correct_pred(
        pred_cell_type, pred_cell_type_mismatch_df, pred_correct_pred_counter)

    st.pyplot(fig)
    st.button("Voir d'autres", key=2, on_click=increment_counter,
                  args=(pred_correct_pred_counter_key,))

    st.markdown("## Images externes au jeu de données d'entraînement")

    cell_type = st.selectbox(
        "Type cellulaire", CLASSES)
    url = st.selectbox("Images", urls_by_cell_type[cell_type])

    url_img = get_image_by_url(url)

    w, h = url_img.size
    max_width = 800

    url_img = url_img.convert('L').resize((max_width, h * max_width // w))

    cropped_img = st_cropper(
        url_img, realtime_update=True, box_color='black', aspect_ratio=(1, 1))

    cropped_img = cropped_img.resize((size, size))

    img_data = np.array(cropped_img).reshape(1, size**2)
    prediction = model.predict(img_data)
    st.write("Type cellulaire prédit :", prediction[0])

    st.markdown("## Prédictions de vos images")

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

    st.markdown("# Prédictions avec une Random Forest")

    size = st.selectbox("Taille des images en entrée du modèle :", [
                        200, 100, 70, 50, 30], index=2)

    st.markdown("## Performances générales sur la base d'apprentissage")

    model_name = 'rfc_'+str(size)
    image_path = get_figure_path(
        'rapport_classification_'+model_name, extension='png')
    image = get_image(image_path)

    st.image(image)

    st.markdown("## Prédictions sur la base d'apprentissage")

    model = load_ml_model('data/ml_models/'+model_name+'.joblib')

    pred_counter_key2 = f'prediction_counter_2'

    init_session_states(pred_counter_key2)

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

    pred_counter = st.session_state[pred_counter_key2]
    fig, pred_to_write = predict_image(pred_counter)
    st.write("Type cellulaire prédit :", pred_to_write)
    st.pyplot(fig)

    st.button(
        "Prédire une autre image",
        key=1,
        on_click=increment_counter,
        args=(pred_counter_key2,))

    st.markdown("## Etude des erreurs faites par le modèle")
    st.markdown("### Répartition")

    fig, mismatch_df = plot_mismatch_distribution(model_name)
    st.write(
        f"Il y a en tout {len(mismatch_df)} images mal classées pour ce modèle.")
    st.pyplot(fig)

    st.markdown("### Visualisation")
    true_cell_type = st.selectbox("Type cellulaire réel:", CLASSES)
    cell_type_mismatch_df = mismatch_df[mismatch_df.true_cell_type ==
                                        true_cell_type]

    st.markdown(f"##### Exemple de {true_cell_type}s correctement prédits.")

    correct_pred_counter_key = f"{model_name}_correct_pred_counter_key"
    init_session_states(correct_pred_counter_key)

    correct_pred_counter = st.session_state[correct_pred_counter_key]
    fig = plot_correct_pred(
        true_cell_type, cell_type_mismatch_df, correct_pred_counter, size)

    st.pyplot(fig)
    st.button("Voir d'autres", on_click=increment_counter,
              args=(correct_pred_counter_key,))
    pred_cell_type = st.selectbox(
        "Type cellulaire prédit:", cell_type_mismatch_df.predicted_cell_type.unique())

    pred_cell_type_mismatch_df = cell_type_mismatch_df[
        cell_type_mismatch_df.predicted_cell_type == pred_cell_type].reset_index(drop=True)

    l = len(pred_cell_type_mismatch_df)
    st.write(
        f"Le type cellulaire {true_cell_type} est confondu {l} fois avec le type cellulaire {pred_cell_type}.")

    fig = plot_pred_compare_with_truth(pred_cell_type_mismatch_df, size=size)
    st.pyplot(fig)
    
    st.markdown(
            f"##### Exemple de {pred_cell_type} correctement prédits.")
    pred_correct_pred_counter_key = f"{model_name}_pred_correct_pred_counter_key"

    init_session_states(pred_correct_pred_counter_key)

    pred_correct_pred_counter = st.session_state[pred_correct_pred_counter_key]
    fig = plot_correct_pred(
        pred_cell_type, pred_cell_type_mismatch_df, pred_correct_pred_counter)

    st.pyplot(fig)
    st.button("Voir d'autres", key=2, on_click=increment_counter,
              args=(pred_correct_pred_counter_key,))

    st.markdown("## Images externes au jeu de données d'entraînement")

    cell_type = st.selectbox(
        "Type cellulaire", CLASSES)
    url = st.selectbox("Images", urls_by_cell_type[cell_type])

    url_img = get_image_by_url(url)

    w, h = url_img.size
    max_width = 800

    url_img = url_img.convert('L').resize((max_width, h * max_width // w))

    cropped_img = st_cropper(
        url_img, realtime_update=True, box_color='black', aspect_ratio=(1, 1))

    cropped_img = cropped_img.resize((size, size))

    img_data = np.array(cropped_img).reshape(1, size**2)
    prediction = model.predict(img_data)
    st.write("Type cellulaire prédit :", prediction[0])

    st.markdown("## Prédictions de vos images")

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
    "Random Forest": section_3,
    "Support Vector Machine": section_2,
}

selected_page = st.sidebar.selectbox(
    "Section : ", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
