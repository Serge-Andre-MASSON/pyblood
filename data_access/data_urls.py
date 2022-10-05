from urllib.request import urlopen
from PIL import Image
import streamlit as st


urls_by_cell_type = {'basophil': [
    "https://imagebank.hematology.org/getimagebyid/60504?size=3",
    "https://imagebank.hematology.org/getimagebyid/60505?size=3",
    "http://medcell.org/histology/blood_bone_marrow_lab/images/basophil.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Basophile-9.JPG/375px-Basophile-9.JPG",
],
    "eosinophil": [
        "http://imagebank.hematology.org/getimagebyid/60934?size=3",
        "http://imagebank.hematology.org/getimagebyid/60933?size=2",
        "http://www.hkimls.org/eos1.jpg",
        "http://www.hkimls.org/eos2.jpg"
],
    'erythroblast': [
        "https://www.cellavision.com/images/Hematopoiesis/Erythropoiesis/PolychromaticErythroblast/POLYCHROMATICERYTHROBLAST1.jpg",
        "https://www.cellavision.com/images/Hematopoiesis/Erythropoiesis/OrthochromaticErythroblast/ORTHROCHROMATICERYTHROBLAST1.jpg",
],
    'ig': ["A compléter"],
    'lymphocyte': ["A compléter"],
    'monocyte': ["A compléter"],
    'neutrophil': [
        "http://imagebank.hematology.org/getimagebyid/61935?size=2"
],
    'platelet': ["A compléter"]}


@st.cache
def get_image_by_url(url):
    with Image.open(urlopen(url)) as img:
        img = img.copy()
    return img
