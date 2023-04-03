#imports
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from os import listdir
from PIL import Image, ImageOps
from io import BytesIO
import plotly.express as px
import plotly.graph_objs as go

import tensorflow as tf
from tensorflow.keras.models import load_model

import urllib.request
import requests
import base64
import os
import glob
import matplotlib.image as mpimg
import pathlib
#------------------------------------------------------------------------------------------------------------------------------------------
# Overall page configuration
st.set_page_config(page_title="BCC", page_icon=":drop_of_blood:", layout="centered", initial_sidebar_state="auto", menu_items=None)

#------------------------------------------------------------------------------------------------------------------------------------------
#streamlit run "C:\Users\User\Desktop\streamlit\23_blood_cells_streamlit_app.py"

#------------------------------------------------------------------------------------------------------------------------------------------
# open images to display (maybe add them to corresponding section for clarity/structure
st.cache_data
def open_image(img):
    return Image.open(img)

img_home_01 = open_image('images/Blood_cell_examples.png')
img_EDA_01 = open_image('images/Image_size.png')
img_EDA_02 = open_image('images/RGB_dist.png')
img_EDA_03 = open_image('images/Grey_dist.png')
Analysis_01 = open_image('images/RESNET_ft_LossVal.png')
Analysis_02 = open_image('images/RESNET_ft_f1.png')
Analysis_04_mix = open_image('images/RESNET_noft_LossVal.png')
Analysis_05_mix = open_image('images/RESNET_noft_F1.png')
Analysis_06_ft_res = open_image('images/RESNET_confusion_matrix.png')
#Analysis_06_mix = open_image('images/Analysis_06_mix.png')
Analysis_07_Amri = open_image('images/Analysis_07_Amri.png')
Analysis_08_Amri = open_image('images/Analysis_08_Amri.png')
Analysis_09_Amri = open_image('images/analysis_09_Amri.png')
cell_01 = open_image('images/M_Bas.jpg')
cell_02 = open_image('images/M_Eos.jpg')
cell_03 = open_image('images/M_Er.jpg')
cell_04 = open_image('images/M_ig.jpg')
cell_05 = open_image('images/M_LT.jpg')
cell_06 = open_image('images/M_Mon.jpg')
cell_07 = open_image('images/M_Neu.jpg')
cell_08 = open_image('images/x.png')
cell_09 = open_image('images/B_Bas.jpg')
cell_10 = open_image('images/B_Eos.jpg')
cell_11 = open_image('images/B_Er.jpg')
cell_12 = open_image('images/B_ig.jpg')
cell_13 = open_image('images/B_LT.jpg')
cell_14 = open_image('images/B_Mon.jpg')
cell_15 = open_image('images/B_Neu.jpg')
cell_16 = open_image('images/B_Plat.jpg')
cell_17 = open_image('images/Ra_Bas.jpg')
cell_18 = open_image('images/Ra_Eos.jpg')
cell_19 = open_image('images/x.png')
cell_20 = open_image('images/x.png')
cell_21 = open_image('images/Ra_LT.jpg')
cell_22 = open_image('images/Ra_Mon.jpg')
cell_23 = open_image('images/Ra_Neu.jpg')
cell_24 = open_image('images/x.png')
hema = open_image('images/Hematopoiesis.jpg')
#------------------------------------------------------------------------------------------------------------------------------------------
# Title of the Page
Header = st.container()
with Header:
    st.title('Automatic Blood Cell Recognition')

# Horizontal menu
selected = option_menu(None, ["Introduction", "E.D.A.", "Modelisation", 'Prediction', 'Perspectives', 'About'],
    icons=["house-door", "bar-chart", "wrench", 'upload', 'search', 'info-circle'],
    menu_icon="droplet", default_index=0, orientation="horizontal")

#------------------------------------------------------------------------------------------------------------------------------------------
# Section Home
if selected == 'Introduction':
    st.header('Introduction')

    st.cache_data
    file_ = open("red-blood-cells-national-geographic.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.cache_data
    st.markdown(
    f'<div style="text-align:center;"><img src="data:image/gif;base64,{data_url}" alt="cat gif"></div>',
    unsafe_allow_html=True,
    )

    st.write("\n\n")


    st.markdown('''
    Blood is a body fluid which flows in the human circulation system and has important functions, such as the supplement of necessary
    substances such as nutrients and oxygen to cells, removing waste and immune defense.

    By the change of their blood components in blood count many diseases can be discovered as well as their severity,
    because of that blood is one of the most examined body fluid in the medical laboratory.

    Especially for hematological diseases, the analysis of the morphology of blood is well known and used in form of blood smear review.
    However, to detect morphological differences between distinct types of normal and abnormal peripheral blood cells, it requires experience,
    skills and time.
    Therefore, it is very helpful for hematological diagnosis the use of automatic blood cell recognition system.

    The main object of this project is to develop a deep learning models to recognize different types of blood cells.
    In general blood cells can be divided into erythrocytes known as red blood cells , leukocytes known as white blood cells and the cell fragments
    called platelets or thrombocytes.
    In this study the focus lies on erythroblasts which are an early stage of erythrocytes and the subdivision of leukocytes such as neutrophils,
    basophils, eosinophils, monocytes ,lymphocytes and immature granulocytes (IG) and the as mentioned above, platelets.\n''')

    # image blood cells
    st.image(img_home_01, caption = 'The different types of blood cells to classify')

    #horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("*The data which enabled this project was derived from three different sources. The entire data is publicly available:*")

    st.markdown("""<div style="color:#696969">
    <ul>
        <li><b>Barcelona:</b> A dataset of microscopic peripheral blood cell images for development of automatic recognition systems, 2020 -
            <a href="https://data.mendeley.com/datasets/snkd93bnjr/1">https://data.mendeley.com/datasets/snkd93bnjr/1</a></li>
        <li><b>Munich:</b> A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Contols (AML-Cytomorhology LMU), 2022 -
            <a href="https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7">
            https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7</a></li>
        <li><b>Raabin:</b> A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm, 2022 -
            <a href="http://dl.raabindata.com/Leukemia_Data/ALL/L1/">http://dl.raabindata.com/Leukemia_Data/ALL/L1/</a></li>
    </ul>
    </div>""", unsafe_allow_html=True)


#------------------------------------------------------------------------------------------------------------------------------------------
#Section EDA
if selected == 'E.D.A.':
    st.header('Exploratory Data Analysis')
    st.markdown(
        """
        Three open source datasets were used to achieve this project's objective. In total they contained ~52,000 images of blood cells. The classification into one of
        eight blood cell types was the target of this project’s model.
        """
    )
    with st.expander("Further information on the datasets"):
        st.subheader('Barcelona')
        st.markdown(
            """
            The first dataset was acquired using the analyzer CellaVision DM96 in the Core Laboratory at the Hospital Clinic of Barcelona. It is organized into eight
            different groups: neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (IG; includung promyelocytes, myelocytes, and metamyelocytes),
            erythroblasts and platelets or thrombocytes. The original image size was 360 × 363 pixels, in format .jpg, and they were annotated by expert clinical pathologists.
            The images were captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the time of blood collection.
            """
        )
        st.subheader('Raabin')
        st.markdown(
            """
            The second dataset is called Raabin-WBC. It is a collection of images from the Razi Hospital in Rasht, Gholhak Laboratory, Shahr-e-Qods Laboratory, and
            Takht-e Tavous Laboratory in Tehran.
            """
        )
        st.subheader('Munich')
        st.markdown(
            """
            The third was the Munich AML Morphology dataset which contained images taken from peripheral blood smears of 100 patients diagnosed with Acute Myeloid Leukemia
            at Munich University Hospital between 2014 and 2017, as well as 100 patients without signs of hematological malignancy with an M8 digital microscope.
            """
        )
        st.subheader('Classes')
        st.markdown(
            """
            Depending on the source, the number of different blood cell classes varied between 8 to 13 classes. Based on the provided descriptions for each dataset and their
            classes, it was decided to merge them into a total of the 8 classes as described for the Barcelona dataset: Neutrophils, eosinophils, basophils, lymphocytes,
            monocytes, immature granulocytes (promyelocytes, myelocytes, and metamyelocytes), erythroblasts and platelets or thrombocytes.
            """
        )

    # Display the dataframe
    st.markdown(
    """
    An extract of the combined dataset with additional features gathered from the files:
    """
    )
    st.cache_data
    df = pd.read_csv("dataframe_eda.csv", index_col=0)

    with st.expander("Show dataset:"):
        st.dataframe(df)

    st.subheader('Image sizes')
    st.markdown("""
    The image size varied between the different datasets, as displayed in the following scatter plot. They were all resized to 360x360 for the
                continuous process.
                """)

   # Create scatterplot with Plotly
    fig = px.scatter(df, x='Width', y='Height', color='Origin', size='Height', symbol='Origin',
                     hover_data={'Shape': True, 'Luminosity' : True, 'Brightness' : True},
                     hover_name="Origin")

    # Set axis labels and title
    fig.update_xaxes(title='Width', showgrid=True)
    fig.update_yaxes(title='Height')
    fig.update_layout(title='Original image resolution', title_font_size=18,
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Brightness')
    st.markdown("""
    The brightness is calculated by the RGB pixel distribution, which show different characteristics according to the classes.
    """)
    st.image(img_EDA_02, caption = 'RGB pixel distribution of the images per class')

    st.write('\n')

    st.subheader('Luminance')
    st.markdown("""
    The luminance is calculated by the greyscale pixel distribution.
                """)
    st.image(img_EDA_03, caption = 'Greyscale pixel distribution of the images per class')


    st.subheader('UMAP')
    st.markdown("""
    The plot of the dimension reduction trough Uniform Manifold Approximation and Projection (UMAP) shows that the images tend to be clustered according to their originating
    dataset instead of the blood cell types. The only class that clearly visible are platelets which are only represented in one dataset.
        """)

    # Load the HTML file
    html_file = open('UMAP_final.html', 'r', encoding='utf-8')
    source_code = html_file.read()

    # Display the HTML file
    components.html(source_code, height=1000, width=1000, scrolling=True)

    st.markdown("""
    The following sample of images sorted by classes and origin as far as available visualizes different stainings and exposures according to their sources. Furthermore
    some images from different blood cell types show high similaritys.
        """)

    with st.container():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.write("<p style='font-size:14px;text-align:center'>Basophil</p>", unsafe_allow_html=True)
        col2.write("<p style='font-size:14px;text-align:center'>Eosinophil</p>", unsafe_allow_html=True)
        col3.write("<p style='font-size:14px;text-align:center'>Erythroblast</p>", unsafe_allow_html=True)
        col4.write("<p style='font-size:14px;text-align:center'>IG</p>", unsafe_allow_html=True)
        col5.write("<p style='font-size:14px;text-align:center'>Lymphocyte</p>", unsafe_allow_html=True)
        col6.write("<p style='font-size:14px;text-align:center'>Monocyte</p>", unsafe_allow_html=True)
        col7.write("<p style='font-size:14px;text-align:center'>Neutrophil</p>", unsafe_allow_html=True)
        col8.write("<p style='font-size:14px;text-align:center'>Platelet</p>", unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.image(cell_01, use_column_width=True, caption = 'BAS, Munich')
        col2.image(cell_02, use_column_width=True, caption = 'EOS, Munich')
        col3.image(cell_03, use_column_width=True, caption = 'ERY,\u200A\u200A\u200A Munich')
        col4.image(cell_04, use_column_width=True, caption = '\u200A\u200A\u200AIG,\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A Munich')
        col5.image(cell_05, use_column_width=True, caption = '\u200A\u200A\u200ALT,\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A Munich')
        col6.image(cell_06, use_column_width=True, caption = 'MON, Munich')
        col7.image(cell_07, use_column_width=True, caption = 'NEU, Munich')

    with st.container():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.image(cell_09, use_column_width=True, caption = 'BAS, Barcelona')
        col2.image(cell_10, use_column_width=True, caption = 'EOS, Barcelona')
        col3.image(cell_11, use_column_width=True, caption = 'ERY, Barcelona')
        col4.image(cell_12, use_column_width=True, caption = 'IG, Barcelona')
        col5.image(cell_13, use_column_width=True, caption = 'LT, Barcelona')
        col6.image(cell_14, use_column_width=True, caption = 'MON, Barcelona')
        col7.image(cell_15, use_column_width=True, caption = 'NEU, Barcelona')
        col8.image(cell_16, use_column_width=True, caption = 'Platelet, Barcelona')

    with st.container():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.image(cell_17, use_column_width=True, caption = 'BAS, Raabin')
        col2.image(cell_18, use_column_width=True, caption = 'EOS, Raabin')
        col5.image(cell_21, use_column_width=True, caption = '\u200A\u200ALT,\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A\u200A Raabin')
        col6.image(cell_22, use_column_width=True, caption = 'MON, Raabin')
        col7.image(cell_23, use_column_width=True, caption = 'NEU, Raabin')
#------------------------------------------------------------------------------------------------------------------------------------------
#Section Models
if selected == 'Modelisation':
    st.header('Modelisation')
    st.markdown('In the following we present the models that achieved the best prediction results:')

    with st.expander("Further information about the modelisation process"):
        st.subheader('First steps')
        st.markdown(
            """
            We started with four pre-trained models ResNet50V2, VGG16, MobileNetV2 and Xception. Without notable image preprocessing or modification
            of layers, hyper-parameters and an imbalanced dataset, the resulting accuracies remained close to random (~12,5% F1). We also faced memory
            issues while working with the entire dataset of ~52.000 images.
            """
        )
        st.subheader('Subsample')
        st.markdown(
            """
            A subsample was created to solve imbalance and memory issues. Based on the class with the smallest representation (Basophil, n=1598), a total
            number of 12784 images was extracted, with each class being evenly represented. This was done using pandas' groupby and sample method. The
            subsample was distributed to every member of the group to stay comparable during modelisation.
            """
        )
        st.subheader('Image Augmentation')
        st.markdown(
            """
            Image augmentation can be usefull to train classification models with small datasets and reduce overfitting. In this case it didn’t. The
            classical ImageDataGenerators resulted in continuously higher validation scores compared to training scores, as well as
            longer runtimes. Considering that blood cell images tend to be recorded in standardized environments with similar methodologies, it was hypothesized
            that too much data augmentation would actually decrease the model's performance. Reducing the image augmentation to horizontal & vertical flips, as well
            as adding random rotations thorugh an augmentation layer combined with re-thinking the layer architecture resulted in the first model hitting above
            an 80% F1-score. Regarding the fact, that the most important information of the image (the blood cell to classify) tends to be in the center of the
            image, surrounded by non-essential red blood cells, it was hypothesized that center crop augmentation would be beneficial. It increased the initial F1 score
            to ~88%.
            """
        )
    st.subheader('ResNet50V2 as base model')
    st.markdown(
        """
        The following two models have been build on a ResNet50V2 base model.
        """)
    
    st.subheader('Simple model')
    st.markdown(
        """
        - **_Image augmentation:_** \u200A horizontal & vertical flips, random rotations and center crop augmentation
        - **_Layer architecture:_** \u200A global average pooling layer, no dropout layers, finishing with a flattened layer and a dense layer with
        a high number of units (before the output layer)
        - F1-score: 91%
        """)
    col1, col2 = st.columns(2)
    col1.image(Analysis_04_mix, use_column_width=True, caption = 'ResNet50V2 Loss')
    col2.image(Analysis_05_mix, use_column_width=True, caption = 'ResNet50V2 Accuracy')

    st.markdown(
        """
        **With fine-tuning:**
        - The last (5th) Conv-block set to be trainable
        - This resulted in over 15 million trainable parameters compared to the initial 164.568 parameters
        - F1-score: 98%
        """)

    col1, col2 = st.columns(2)

    col1.image(Analysis_01, use_column_width=True, caption = 'ResNet50V2 with fine tuning Loss')
    col2.image(Analysis_02, use_column_width=True, caption = 'ResNet50V2 with fine tuning Accuracy')
    
    st.image(Analysis_06_ft_res, caption = 'ResNet50V2 Confusion Matrix')

    st.subheader('Mixed inputs')
    st.markdown(
        """
        - Same architecture and fine-tuning as the previous model
        - The features luminosity and brightness were used as additional numerical input next to the image arrays
        - F1-score: 97.3%
        """)

  

    # st.image(Analysis_06_mix, caption = 'Confusion Matrix')

    st.subheader('VGG16 as base model')
    st.markdown(
        """
        - **_Image augmentation:_** \u200A horizontal & vertical flips and random rotations
        - **_Layer architecture:_** \u200A one global average pooling layer, two large Dense layers followed by a slight dropout layer
        - F1-score: 86%
        """)
    st.markdown(
        """
        **With fine-tuning:**
        - The last 5 layers were set to trainable
        - This resulted in close to 13 million trainable parameters compared to the initial 6.501.816 parameters.
        - F1-score: 96%
        """)

    col1, col2 = st.columns(2)
    col1.image(Analysis_07_Amri, use_column_width=True, caption = 'VGG16 Loss')
    col2.image(Analysis_08_Amri, use_column_width=True, caption = 'VGG16 Accuracy')

    st.image(Analysis_09_Amri, caption = 'VGG16 Confusion Matrix')
#------------------------------------------------------------------------------------------------------------------------------------------
#Section Prediction

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Necessary function and variables
RES_MODEL = "models/Best_model_ft_5th_layer.h5"
VGG_MODEL = "models/vgg16_augmented_model.h5"
IMG_SIZE = (360,360)

CLASS_LABELS = ['Basophil',
                'Eosinophil',
                'Erythroblast',
                'Immature granulocytes',
                'Lymphocyte',
                'Monocyte',
                'Neutrophil',
                'Platelet']

#function to load model
@st.cache_resource
def load_dl_model(model_choice):
    if not os.path.isfile(model_choice):
        urllib.request.urlretrieve(f"https://github.com/RoadieHouse/Blood-Cell-Classification/blob/main/{model_choice}", model_choice[:7])
    return tf.keras.models.load_model(model_choice)

# Calculate f1 score
def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall

# Preprocess image
def preprocess_image(image):
    if image is not None:
        image = ImageOps.fit(image, IMG_SIZE, Image.BICUBIC)
        image = tf.keras.preprocessing.image.img_to_array(image)
        #image = tf.keras.applications.resnet_v2.preprocess_input(image)
        return image

# Function to make predictions
def predict(image):
    if image is not None:
        image = preprocess_image(image)
        predictions = model.predict(tf.expand_dims(image, axis=0))[0]
        predicted_class = CLASS_LABELS[predictions.argmax()]
        confidence = predictions.max()
        return predicted_class, confidence

# list all available images to make predicitions on (no images uploaded so far right?)
def list_images(directory, file_type):
    if(file_type != 'Please make selection'):
        directory += file_type
        st.write(directory)
        files = listdir('https://github.com/RoadieHouse/Blood-Cell-Classification/blob/main/' + directory)
        st.write(files)
        #files[0] = "Select from list"
        #images = []
        #for img_path in glob.glob(directory + '/*'):
            #images.append(mpimg.imread(img_path))

        #file = st.selectbox("Pick an image to test",images)

 
        #st.write(file)
        return file
    else:
        return Null

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if selected == 'Prediction':

    st.header('Prediction')
    st.subheader("Here you can choose a model to classify a blood cell image")

    model_for_prediction = st.selectbox("Select a model", ["Select a model:", "Resnet50V2", "VGG16"])
    if model_for_prediction == "Select a model":
        st.info("A model has to be selected to make a prediction.")
    else:
        l_col, r_col = st.columns(2)

        with l_col:
            image_file = st.file_uploader("Upload an image to classify:", type=["jpg", "jpeg", "png", "tiff"])

        with r_col:
            selected_class = st.selectbox("Select a class:", [*CLASS_LABELS])
            
            directory = '/images/'
            directory += selected_class
            #st.write(directory)
            #st.write(files)
            #selected_file = list_images(directory, selected_class)
            #image_file = directory + selected_class + '/' + selected_file
            #image_file = 'images/basophil/BAS_0016.tiff'

            st.write(pathlib.PurePath('./images')
            

        if image_file is not None:
            image = open_image(image_file)
            st.image(image, caption="Uploaded Image", width = 180)

        else:
            st.info("Please upload an image to classify or choose one from the dropdown manu on the right")
            #something with selected classes
            #image = ...

        if (st.button("Predict")):

            if model_for_prediction == "Resnet50V2":

                #Create a dictionary mapping the function name to the function object
                custom_objects = {'f1': f1}

                # Load the Keras model using custom_object_scope
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = load_dl_model(RES_MODEL)

            if model_for_prediction == "VGG16":
                model = load_dl_model(VGG_MODEL)


            #st.image(image, caption="Uploaded Image", width = 180)
            predicted_class, confidence = predict(image)

            col1, col2 = st.columns(2)
            with col1:
                st.title("Predicted class:")
                st.subheader(f"{predicted_class}")
            with col2:
                st.title("Confidence score:")
                conf_percent = confidence * 100
                if conf_percent > 50:
                    #st.markdown(f"<p style=color: green; font-size: 20px;>{conf_percent}%</span>", unsafe_allow_html=True)
                    st.subheader(f":green[{conf_percent:.2f}%]")
                else:
                    st.subheader(f":red[{conf_percent:.2f}%]")



            st.write("")
            st.write("\n\n Additional information:")

            # Display additional information about the predicted class
            if predicted_class == "Eosinophil":
                st.info("Eosinophils are a type of white blood cell involved in the immune response to parasites and allergies.")
            elif predicted_class == "Lymphocyte":
                st.info("Lymphocytes are a type of white blood cell involved in the immune response to infections and cancer.")
            elif predicted_class == "Monocyte":
                st.info("Monocytes are a type of white blood cell involved in the immune response to infections and inflammation.")
            elif predicted_class == "Neutrophil":
                st.info("Neutrophils are a type of white blood cell involved in the immune response to bacterial and fungal infections.")
            elif predicted_class == "Immature granulocytes":
                st.info("Immature granulocytes, including promyelocytes, myelocytes, and metamyelocytes, are early-stage white blood cells that are typically elevated in response to acute bacterial infections and inflammatory disorders.")
            elif predicted_class == "Basophils":
                st.info("Basophils are a type of white blood cell involved in the immune response against parasites and are also involved in the inflammatory response.")
            elif predicted_class == "Platelet":
                st.info("Platelets are small, colorless cell fragments in the blood that play a crucial role in blood clotting and the prevention of excessive bleeding.")
            elif predicted_class == "Erythroblast":
                st.info("Erythroblasts are immature red blood cells that are involved in the production of hemoglobin and the transportation of oxygen throughout the body.")

    # Add some padding and styling elements to the selectbox and file uploader
    st.markdown('<style>div[role="listbox"] > div:nth-child(1) {padding: 10px; font-family: Arial, sans-serif;}</style>', unsafe_allow_html=True)
    st.markdown('<style>.css-1aya9p5 {font-family: Arial, sans-serif;}</style>', unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------------------------------------
#Section Perspectives
if selected == 'Perspectives':
    st.header('Perspectives')
    st.markdown("""
        The role of machine learning methods in intelligent medical diagnostics is becoming more and more present these days.
        And deep neural networks are revolutionizing the medical diagnosis process rapidly.

         - in reality,there is a wider spectrum of blood cell types, regarding to subcategories of immature granulocytes and other early stages of blood cell.
         In this project the focus was to detect 8 different blood cells types.""")

    st.image(hema, caption = 'Hematopoiesis')
   
    st.markdown("""
        - the training data set should be as diverse and precise as possible to classify the blood cells.

        - different sources can considerly change the outcome of images, like the different medical devices, microscope and camera,
        and the method of processing the blood cells, the use of stain.

        - the dataset can be used to recognize the blood cell type and trained further to classify other types of abnormal cells.
        """)
#------------------------------------------------------------------------------------------------------------------------------------------
#Section About
if selected == 'About':
    st.header('About')
    st.markdown('This machine learning project was part of Datascientest International Class at University of Paris La Sorbonne.')
    st.header('Contributors')
    st.write('Amritha Kalluvettukuzhiyil  \n Elias Zitterbarth  \n Daniela Hummel  \n Lilli Krizek')
    st.subheader('Image References')
    st.markdown("""
    - https://tenor.com/de/view/red-blood-cells-national-geographic-arteries-blood-flow-world-heart-day-gif-18613531
    - https://en.wikipedia.org/wiki/Haematopoiesis
     """)
