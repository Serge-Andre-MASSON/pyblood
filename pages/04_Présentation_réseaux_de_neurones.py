import streamlit as st

from data_access.data_access import load_pickle
from data_access.data_paths import get_pbc_dataset_infos_paths
from keras.applications.vgg16 import VGG16
from keras.layers.preprocessing.image_preprocessing import Rescaling
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.models import load_model

# Présentation du contenu sur la sidebar
st.sidebar.markdown("# Présentation des CNN utilisés")
st.sidebar.write("""Dans cette partie, nous présentons les caractéristiques des divers réseaux de neurones testés ainsi, que leurs performances.
Les résultats présentés sont ceux des transfer learning effectués à partir de chacun des modèles choisis. """)

def section_1():
    st.header("Modèle de transfer learning avec VGG16")
    st.subheader("Paramètres du modèle et résultats")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    #img_shape=(256,256,3)
    #vgg16=VGG16(input_shape=img_shape, weights="imagenet", include_top=False)
    #for layer in vgg16.layers[-4:]:
        #layer.trainable = True
    #normalization_layer = Rescaling(1./255, input_shape=img_shape)
    #model_vgg16=Sequential()
    #model_vgg16.add(normalization_layer)
    #model_vgg16.add(vgg16)
    #model_vgg16.add(Dense(500, activation='relu'))
    #model_vgg16.add(Dropout(0.5))
    #model_vgg16.add(Dense(250, activation='relu'))
    #model_vgg16.add(Dropout(0.5))
    #model_vgg16.add(Flatten())
    #model_vgg16.add(Dense(8, activation='softmax'))
    #model_vgg16.summary(print_fn=lambda x: st.text(x))
    
    # Display Images
 
    # import Image from pillow to open images
    from PIL import Image
    # chemin d'accès aux figures
    dir_img="D:\Bureau\FORMATION DATASCIENCE\PROJET\pyblood\data\\figures"
    # image du résumé du modèle vgg16
    img = Image.open(dir_img + "\\vgg16_summary.jpg")
 
    # display image using streamlit
    # width is used to set the width of an image
    st.image(img, width=800)

    st.write("L’entraînement du modèle est effectué sur seulement 10 epochs mais se stoppe à 6 epochs car l’accuracy n’évolue plus et nous obtenons une accuracy de 95% contre 94% sans fine-tuning avec un temps d’entraînement d’environ 10 minutes. En revanche, nous remarquons un léger overfitting avec ce modèle")
   
    # image du rapport de classification vgg16
    img_2=Image.open(dir_img+"\\rapport_classification_vgg16.jpg")
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    col2.image(img_2, use_column_width=True, width=200)
    
    
    # image de la courbe accuracy de vgg16
    img_3=Image.open(dir_img+'\\courbe_vgg16.jpg')
    col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
    col2.image(img_3, use_column_width=True, width=200)
    

   
    
def section_2():
    st.header("Modèle de transfer learning avec InceptionV3")
    st.subheader("Paramètres du modèle et résultats")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    from PIL import Image
    # image du résumé du modèle inceptionV3
    dir_img="D:\Bureau\FORMATION DATASCIENCE\PROJET\pyblood\data\\figures"
    img_4=Image.open(dir_img+'\\inceptionv3_summary.jpg')
    st.image(img_4, width=800)
    st.write("L’entraînement du modèle est effectué sur seulement 10 epochs et nous obtenons une accuracy de 92% contre 90% sans fine-tuning avec un temps d’entraînement d’environ 10 minutes. En revanche, nous remarquons un overfitting assez important avec ce modèle.")

    # image du rapport de classification inceptionV3
    img_5=Image.open(dir_img+"\\rapport_classification_inceptionv3.jpg")
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    col2.image(img_5, use_column_width=True, width=200)

    # image de la courbe accuracy de inceptionv3
    img_6=Image.open(dir_img+'\\courbe_inceptionv3.jpg')
    col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
    col2.image(img_6, use_column_width=True, width=200)

def section_3():
    st.header("Modèle de transfer learning avec Densenet121")
    st.subheader("Paramètres du modèle et résultats")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    from PIL import Image
    # image du résumé du modèle Densenet121
    dir_img="D:\Bureau\FORMATION DATASCIENCE\PROJET\pyblood\data\\figures"
    img_4=Image.open(dir_img+'\\densenet_summary.jpg')
    st.image(img_4, width=800)
    st.write("L’entraînement du modèle est effectué sur seulement 25 epochs et nous obtenons une accuracy de 98% contre 95% sans fine-tuning avec un temps d’entraînement d’environ 50 minutes.")

    # image du rapport de classification Densenet121
    img_5=Image.open(dir_img+"\\rapport_classification_densenet.jpg")
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    col2.image(img_5, use_column_width=True, width=200)

    # image de la courbe accuracy de Densenet121
    img_6=Image.open(dir_img+'\\courbe_densenet.jpg')
    col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
    col2.image(img_6, use_column_width=True, width=200)

def section_4():
    st.header("Modèle de transfer learning avec Basic Model")
    st.subheader("Paramètres du modèle et résultats")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    from PIL import Image
    # image du résumé du modèle Basic Model
    dir_img="D:\Bureau\FORMATION DATASCIENCE\PROJET\pyblood\data\\figures"
    img_4=Image.open(dir_img+'\\basic_model_summary.jpg')
    st.image(img_4, width=800)
    st.write("L’entraînement du modèle est effectué jusqu’à ce que l’accuracy de validation augmente de moins de 0.1% sur 10 epochs, avec également un mécanisme de réduction du taux d’apprentissage en cas de détection de plateau. Nous obtenons une accuracy de 96% avec un temps d’entraînement d’environ 30 minutes. Nous constatons que le phénomène de divergence précédemment observé se reproduit, mais plus tard, à partir de la 20ème epoch environ. De plus, il est moins prononcé, l’accuracy d’entraînement montant jusqu’aux alentours de 99% tandis que celle de validation plafonne autour de 96% à partir de la 25ème epoch environ.")
    # image du rapport de classification Basic Model
    img_5=Image.open(dir_img+"\\rapport_classification_basic_model.jpg")
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    col2.image(img_5, use_column_width=True, width=200)

    # image de la courbe accuracy de Basic Model
    img_6=Image.open(dir_img+'\\courbe_basic_model.jpg')
    col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
    col2.image(img_6, use_column_width=True, width=200)
    
page_names_to_funcs = {
    "VGG16": section_1,
    "InceptionV3": section_2,
    "DenseNet121": section_3,
    "Basic Model": section_4
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
