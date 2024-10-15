import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import requests
import base64
import os

#------------------------------------------------------------------------------------------------------------------------------------------
# Overall page configuration
st.set_page_config(page_title="BCC", page_icon=":drop_of_blood:", layout="centered", initial_sidebar_state="auto", menu_items=None)

#------------------------------------------------------------------------------------------------------------------------------------------
# Open images to display (maybe add them to corresponding section for clarity/structure
@st.cache_data
def open_image(img):
    return Image.open(img)

img_home_01 = open_image('images/Blood_cell_examples.png')
#img_EDA_01 = open_image('images/Image_size.png')
img_EDA_02 = open_image('images/RGB_dist.png')
img_EDA_03 = open_image('images/Grey_dist.png')
Analysis_01 = open_image('images/RESNET_noft_LossVal.png')
Analysis_02 = open_image('images/RESNET_noft_F1.png')
Analysis_03 = open_image('images/RESNET_ft_LossVal.png')
Analysis_04 = open_image('images/RESNET_ft_f1.png')
Analysis_05 = open_image('images/RESNET_confusion_matrix.png')
Analysis_06 = open_image('images/Analysis_07_Amri.png')
Analysis_07 = open_image('images/Analysis_08_Amri.png')
Analysis_08 = open_image('images/VGG_confusion_matrix.png')
cell_01 = open_image('images/M_Bas.jpg')
cell_02 = open_image('images/M_Eos.jpg')
cell_03 = open_image('images/M_Er.jpg')
cell_04 = open_image('images/M_ig.jpg')
cell_05 = open_image('images/M_LT.jpg')
cell_06 = open_image('images/M_Mon.jpg')
cell_07 = open_image('images/M_Neu.jpg')
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
cell_21 = open_image('images/Ra_LT.jpg')
cell_22 = open_image('images/Ra_Mon.jpg')
cell_23 = open_image('images/Ra_Neu.jpg')
no_image_placeholder = open_image('images/placeholder.jpg')
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
    
    #blood flow GIF
    file_ = open("red-blood-cells-national-geographic.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
    f'<div style="text-align:center;"><img src="data:image/gif;base64,{data_url}" alt="cat gif"></div>',
    unsafe_allow_html=True,
    )
    
    #increase pad
    st.write("\n\n")


    st.markdown('''
    Blood is a vital fluid that circulates through the human body, performing critical functions such as delivering oxygen and nutrients to cells, removing waste products, and supporting the immune system.

    Changes in blood composition, detected through blood counts, can reveal the presence and severity of various diseases. Because of this, blood is one of the most frequently analyzed fluids in medical laboratories.

    For hematological conditions in particular, blood smear reviews—used to assess blood cell morphology—are a well-established diagnostic tool. However, identifying the subtle differences between normal and abnormal peripheral blood cells requires significant expertise, experience, and time. This is where automated blood cell recognition systems become invaluable, providing speed and precision in hematological diagnoses.

    The objective of this project was to develop a deep learning model capable of recognizing different types of blood cells. Blood cells can generally be classified into three main categories: erythrocytes (red blood cells), leukocytes (white blood cells), and thrombocytes (platelets, which are cell fragments). This study focuses on the early-stage erythrocytes known as erythroblasts, as well as specific subtypes of leukocytes, including neutrophils, basophils, eosinophils, monocytes, lymphocytes, immature granulocytes (IG), and the previously mentioned platelets.\n''')

    #image blood cells
    st.image(img_home_01, caption = 'Blood Cell Types')

    #horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    #links to datasets
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
# Section EDA
if selected == 'E.D.A.':
    st.header('Exploratory Data Analysis')
    st.markdown(
        """
        Three open-source datasets were used to achieve the objectives of this project, containing approximately 52,000 images of blood cells in total. The goal of the project was to classify these images into one of eight distinct blood cell types.
        """
    )
    with st.expander("Further information on the datasets"):
        st.subheader('Barcelona')
        st.markdown(
            """
            The first dataset was obtained using the CellaVision DM96 analyzer in the Core Laboratory at the Hospital Clinic of Barcelona. This dataset is organized into eight categories: neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (IG; including promyelocytes, myelocytes, and metamyelocytes), erythroblasts, and platelets (thrombocytes). Each image was 360 × 363 pixels in .jpg format and was carefully annotated by expert clinical pathologists. The images were collected from individuals free of infections, hematologic or oncologic diseases, and not under any pharmacologic treatment at the time of blood collection."""
            )
        st.subheader('Raabin')
        st.markdown(
            """
            The second dataset, called Raabin-WBC, consists of images collected from the Razi Hospital in Rasht, as well as Gholhak, Shahr-e-Qods, and Takht-e Tavous Laboratories in Tehran."""
            )
        st.subheader('Munich')
        st.markdown(
            """
            The third dataset, known as the Munich AML Morphology dataset, includes images taken from peripheral blood smears of 100 patients diagnosed with Acute Myeloid Leukemia (AML) at Munich University Hospital between 2014 and 2017. It also includes images from 100 individuals without hematological malignancies, captured using an M8 digital microscope."""
        )
        st.subheader('Classes')
        st.markdown(
            """
            Depending on the source, the number of blood cell classes varied between 8 and 13. After reviewing the dataset descriptions and classifications, we standardized them into eight classes as described for the Barcelona dataset: neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (promyelocytes, myelocytes, and metamyelocytes), erythroblasts, and platelets (thrombocytes)."""
        )

    #Display the dataframe
    st.markdown(
        """
        An extract of the combined dataset includes additional features gathered from the files:"""
        )
    df = pd.read_csv("dataframe_eda.csv", index_col=0)

    with st.expander("Show dataset:"):
        st.dataframe(df)

    st.subheader('Image sizes')
    st.markdown(
        """
        The image sizes varied between the different datasets, as shown in the accompanying scatter plot. To ensure consistency, all images were resized to 360 × 360 pixels for further processing."""
        )

    #Create scatterplot with Plotly
    fig = px.scatter(df, x='Width', y='Height', color='Origin', size='Height', symbol='Origin',
                     hover_data={'Shape': True, 'Luminosity' : True, 'Brightness' : True},
                     hover_name="Origin")

    #Set axis labels and title
    fig.update_xaxes(title='Width', showgrid=True)
    fig.update_yaxes(title='Height')
    fig.update_layout(title='Original image resolution', title_font_size=18,
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")

    #Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    #Display RGB distibution
    st.subheader('Brightness')
    st.markdown("""
    The brightness is calculated by the RGB pixel distribution, which show different characteristics according to the classes.
    """)
    st.image(img_EDA_02, caption = 'RGB pixel distribution of the images per class')

    st.write('\n')

    #Display Greyscale distribution
    st.subheader('Luminance')
    st.markdown("""
    The luminance is calculated by the greyscale pixel distribution.
                """)
    st.image(img_EDA_03, caption = 'Greyscale pixel distribution of the images per class')


    st.subheader('UMAP')
    st.markdown("""
    The plot of the dimension reduction trough Uniform Manifold Approximation and Projection (UMAP) shows that the images tend to be clustered according to their originating
    dataset instead of the blood cell types. The only class that is clearly visible are platelets which are only represented in one dataset.
        """)

    # Load the HTML UMAP file
    html_file = open('UMAP_final.html', 'r', encoding='utf-8')
    source_code = html_file.read()

    # Display the HTML UMAP file
    components.html(source_code, height=1000, width=1000, scrolling=True)

    st.markdown(
        """
        The following sample of images, sorted by class and origin, highlights the variations in staining techniques and exposure conditions based on the source. Additionally, some images of different blood cell types displayed significant visual similarities."""
        )

        # Create headers for the columns (blood cell types)
    st.write("<p style='text-align:center;font-size:18px'>Blood Cell Type by Origin\n</p>", unsafe_allow_html=True)
    headers = ["","Basophil", "Eosinophil", "Erythroblast", "Imm. Granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]
    columns = st.columns(9)
    for i, header in enumerate(headers):
        with columns[i]:
            st.write(f"<p style='font-size:12px;text-align:center'>{header}</p>", unsafe_allow_html=True)

    # First row - Munich
    cols = st.columns(9)
    cols[0].write(
    "<div style='display: flex; justify-content: center; align-items: center; padding-top: 25px'>"
    "<p style='font-size:12px;text-align:center;'>Munich</p></div>", unsafe_allow_html=True)
    cols[1].image(cell_01, use_column_width=True)
    cols[2].image(cell_02, use_column_width=True)
    cols[3].image(cell_03, use_column_width=True)
    cols[4].image(cell_04, use_column_width=True)
    cols[5].image(cell_05, use_column_width=True)
    cols[6].image(cell_06, use_column_width=True)
    cols[7].image(cell_07, use_column_width=True)
    cols[8].image(no_image_placeholder, use_column_width=True)  # No platelet image

    # Second row - Barcelona
    cols = st.columns(9)
    cols[0].write(
    "<div style='display: flex; justify-content: center; align-items: center; padding-top: 23px'>"
    "<p style='font-size:12px;text-align:center;'>Barcelona</p></div>", unsafe_allow_html=True)
    cols[1].image(cell_09, use_column_width=True)
    cols[2].image(cell_10, use_column_width=True)
    cols[3].image(cell_11, use_column_width=True)
    cols[4].image(cell_12, use_column_width=True)
    cols[5].image(cell_13, use_column_width=True)
    cols[6].image(cell_14, use_column_width=True)
    cols[7].image(cell_15, use_column_width=True)
    cols[8].image(cell_16, use_column_width=True)

    # Third row - Raabin
    cols = st.columns(9)
    cols[0].write(
    "<div style='display: flex; justify-content: center; align-items: center; padding-top: 23px'>"
    "<p style='font-size:12px;text-align:center;'>Raabin</p></div>", unsafe_allow_html=True)
    cols[1].image(cell_17, use_column_width=True)
    cols[2].image(cell_18, use_column_width=True)
    cols[3].image(no_image_placeholder, use_column_width=True)  # No erythroblast image
    cols[4].image(no_image_placeholder, use_column_width=True)  # No IG image
    cols[5].image(cell_21, use_column_width=True)
    cols[6].image(cell_22, use_column_width=True)
    cols[7].image(cell_23, use_column_width=True)
    cols[8].image(no_image_placeholder, use_column_width=True)  # No platelet image
        
#------------------------------------------------------------------------------------------------------------------------------------------
# Section Models
if selected == 'Modelisation':
    st.header('Modelisation')
    st.markdown('This section presents the models that achieved the best prediction results:')

    # Initial information expander
    with st.expander("Detailed insights into the modelisation process"):
        st.subheader('Initial Approach')
        st.markdown(
            """
            The journey began with four pre-trained models: ResNet50V2, VGG16, MobileNetV2, and Xception. Initially, without substantial image preprocessing 
            or modifications to layers and hyperparameters, and working with an imbalanced dataset, the results were disappointing. Accuracies hovered around 
            12.5% F1-score, barely above random chance. Additionally, there were issues with memory constraints when attempting to process the entire dataset of 
            approximately 52,000 images.
            """
        )
        st.subheader('Strategic Subsampling')
        st.markdown(
            """
            To address both class imbalance and memory issues, a balanced subsample was created. The least represented class (Basophil, n=1,598) was used as the 
            baseline to extract an equal number of images from each class. This resulted in a total of 12,784 images, with each class equally represented. 
            Pandas' groupby and sample methods were utilized for this process.
            """
        )
        st.subheader('Refined Image Augmentation')
        st.markdown(
            """
            While image augmentation is often beneficial for training classification models with limited datasets and reducing overfitting, the experience proved 
            unique. Traditional ImageDataGenerators led to consistently higher validation scores compared to training scores and increased processing times. 

            It was hypothesized that blood cell images, typically captured in standardized environments using similar methodologies, might not benefit from extensive 
            augmentation. In fact, too much augmentation could potentially decrease model performance.

            The breakthrough came when the approach was redefined:
            1. Reduced image augmentation to only use horizontal and vertical flips.
            2. Introduction of random rotations through a dedicated augmentation layer.
            3. Redesigned layer architecture.

            This optimized strategy resulted in the first model achieving an F1-score above 80%. 

            Further analysis revealed that the most crucial information (the blood cell to be classified) typically occupies the center of the image, surrounded by 
            non-essential red blood cells. Based on this observation, center crop augmentation was implemented, which boosted the initial F1-score to approximately 88%.

            These refinements demonstrate the importance of domain-specific knowledge in machine learning applications and the value of iterative improvement in 
            model development.
            """
        )
        
    #Resnet model performance
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
    col1.image(Analysis_01, use_column_width=True, caption = 'ResNet50V2 Loss')
    col2.image(Analysis_02, use_column_width=True, caption = 'ResNet50V2 Accuracy')

    st.markdown(
        """
        **With fine-tuning:**
        - The last (5th) Conv-block set to be trainable
        - This resulted in over 15 million trainable parameters compared to the initial 164.568 parameters
        - F1-score: 98%
        """)

    col1, col2 = st.columns(2)
    col1.image(Analysis_03, use_column_width=True, caption = 'ResNet50V2 with fine tuning Loss')
    col2.image(Analysis_04, use_column_width=True, caption = 'ResNet50V2 with fine tuning Accuracy')
    
    st.image(Analysis_05, caption = 'ResNet50V2 Confusion Matrix')

    #Mixed input model
    st.subheader('Mixed inputs')
    st.markdown(
        """
        - Same architecture and fine-tuning as the previous model
        - The features luminosity and brightness were used as additional numerical input next to the image arrays
        - F1-score: 97.3%
        """)
    
    #VGG model
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
    col1.image(Analysis_06, use_column_width=True, caption = 'VGG16 Loss')
    col2.image(Analysis_07, use_column_width=True, caption = 'VGG16 Accuracy')

    st.image(Analysis_08, caption = 'VGG16 Confusion Matrix')
#------------------------------------------------------------------------------------------------------------------------------------------
#Section Prediction

def get_api_endpoint(model_choice: str) -> str:
        if model_choice == "Resnet50V2":
            return st.secrets["RESNET_MODEL_API"]
        elif model_choice == "VGG16":
            return st.secrets["VGG_MODEL_API"]
        else:
            raise ValueError("Invalid model choice")
    
def predict_image(image_file, api_endpoint: str):
    files = {"file": ("image.jpg", image_file, "image/jpeg")}
    try:
        response = requests.post(api_endpoint, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the API: {str(e)}")
    except ValueError as e:
        st.error(f"Error decoding API response: {str(e)}")
    return None

def display_prediction_results(prediction_data: dict):
    label = prediction_data['predicted_label']
    confidence = prediction_data['confidence']
    color = "green" if confidence >= 0.5 else "red"
    st.markdown(f"""## Predicted Label:\n#### {label} (Confidence: <span style="color:{color}; font-weight:bold;">{confidence:.2f}</span>)
    """, unsafe_allow_html=True)
    
    st.write("")
    
    st.write("\n\n Additional information:")
    
        # Display additional information about the predicted class
    if label == "Eosinophil":
        st.info("Eosinophils are a type of white blood cell involved in the immune response to parasites and allergies.")
    elif label == "Lymphocyte":
        st.info("Lymphocytes are a type of white blood cell involved in the immune response to infections and cancer.")
    elif label == "Monocyte":
        st.info("Monocytes are a type of white blood cell involved in the immune response to infections and inflammation.")
    elif label == "Neutrophil":
        st.info("Neutrophils are a type of white blood cell involved in the immune response to bacterial and fungal infections.")
    elif label == "IG":
        st.info("Immature granulocytes, including promyelocytes, myelocytes, and metamyelocytes, are early-stage white blood cells that are typically elevated in response to acute bacterial infections and inflammatory disorders.")
    elif label == "Basophil":
        st.info("Basophils are a type of white blood cell that play a role in the immune response against parasites and contribute to the inflammatory response.")
    elif label == "Platelet":
        st.info("Platelets are small, colorless cell fragments in the blood that play a crucial role in blood clotting and the prevention of excessive bleeding.")
    elif label == "Erythroblast":
        st.info("Erythroblasts are immature red blood cells that are involved in the production of hemoglobin and the transportation of oxygen throughout the body.")
    
    
    with st.expander("View Detailed Results"):
        st.subheader("Class Probabilities")
        for class_name, prob in prediction_data['class_probabilities'].items():
            col1, col2, col3 = st.columns([2,1,6])  # Adjust column ratios to control spacing
            col1.write(f"{class_name}")
            col2.write(f"{prob:.4f}")
            col3.progress(prob)
        
        fig = go.Figure(data=[go.Bar(
            x=list(prediction_data['class_probabilities'].keys()),
            y=list(prediction_data['class_probabilities'].values()),
            marker_color='rgb(26, 118, 255)'
        )])
        fig.update_layout(
            xaxis_title="Cell Type",
            yaxis_title="Probability",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Header prediction
if selected == 'Prediction':

    st.header('Prediction')
    st.subheader("Choose a model to classify a blood cell image")

    model_choice = st.selectbox("Select a model:", ["Please select", "Resnet50V2", "VGG16"])
    
    if model_choice == "Please select":
        st.info("A model has to be selected to make a prediction.")

    image_file = st.file_uploader("Upload an image to classify:", type=["jpg", "jpeg", "png", "tiff"])
         
    if image_file is not None:
        st.image(image_file, caption="Uploaded Image", width=180)
        
        if st.button("Predict"):
            api_endpoint = get_api_endpoint(model_choice)
            prediction_data = predict_image(image_file, api_endpoint)
            
            if prediction_data and "error" not in prediction_data:
                display_prediction_results(prediction_data)
                
            elif prediction_data and "error" in prediction_data:
                st.error(f"API Error: {prediction_data['error']}")
                if "traceback" in prediction_data:
                    st.error(f"Traceback: {prediction_data['traceback']}")
    else:
        st.warning("Please upload an image before predicting.")
        
#------------------------------------------------------------------------------------------------------------------------------------------
#Section Perspectives
if selected == 'Perspectives':
    st.header('Future Perspectives in Medical Diagnostics')
    st.markdown("""
        Machine learning methods are increasingly playing a crucial role in medical diagnostics. Deep neural networks, in particular, have revolutionized 
        the field, significantly enhancing the accuracy and efficiency of medical diagnoses. As we look to the future, several important considerations emerge:

        1. **Expanding Cell Type Recognition:**
           The project focused on eight distinct blood cell types. However, in reality, there exists a much broader spectrum of blood cells, including:
           - Subcategories of immature granulocytes
           - Various stages of early blood cell development
           - Rare cell types not covered in our current classification system

           Expanding the model to recognize these additional cell types could greatly enhance its clinical utility.
        """)

    st.image(hema, caption='Hematopoiesis: The process of blood cell formation')
   
    st.markdown("""
        2. **Enhancing Dataset Diversity and Precision:**
           To achieve optimal outcomes and high generalizability in blood cell classification (and indeed, any image classification task), training datasets should be:
           - Diverse: Representing a wide range of patient demographics, clinical conditions, and cell morphologies
           - Precise: Accurately labeled and vetted by expert hematologists
           - Comprehensive: Including a large number of samples for each cell type, including rare variants

        3. **Addressing Variability in Imaging Conditions:**
           Several factors can significantly impact the appearance of blood cells in images:
           - Medical devices: Different microscopes and cameras can produce varying image qualities and characteristics
           - Sample processing methods: Variations in blood smear preparation techniques can affect cell appearance
           - Staining procedures: Different staining methods can alter the color and contrast of cell features

           Future models should be robust enough to handle these variations or include preprocessing steps to normalize images.

        4. **Broader Applications:**
           While the current focus is on blood cell classification, the techniques and models developed here have potential applications in other areas of medical imaging, such as:
           - Identifying abnormal cells in other body fluids (e.g., cerebrospinal fluid, pleural fluid)
           - Detecting cellular changes indicative of various diseases or conditions
           - Assisting in the diagnosis of blood-related disorders and hematological malignancies

        As we continue to refine and expand these models, their potential to support and enhance medical diagnostics grows exponentially, promising faster, 
        more accurate, and more accessible health care solutions.
        """)
    
#------------------------------------------------------------------------------------------------------------------------------------------
#Section About
if selected == 'About':
    st.header('About')
    
    st.markdown('This machine learning project was part of DataScientest International Class @[University of Paris La Sorbonne](https://www.sorbonne-universite.fr/en).')
    
    st.header('Contributor:')
    
    st.markdown('''
    Elias Zitterbarth  ( [LinkedIn](https://www.linkedin.com/in/elias-zitterbarth)  &  [GitHub](https://github.com/RoadieHouse) )''', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    #image references
    st.subheader('Image References')
    st.markdown("""<div style="color:#696969">
    <ul>
        <li><a href="https://tenor.com/de/view/red-blood-cells-national-geographic-arteries-blood-flow-world-heart-day-gif-18613531">https://tenor.com/de/view/red-blood-cells-national-geographic-arteries-blood-flow-world-heart-day-gif-18613531</a></li>
        <li><a href="https://en.wikipedia.org/wiki/Haematopoiesis">https://en.wikipedia.org/wiki/Haematopoiesis</a></li>
    </ul>
    </div>""",unsafe_allow_html=True)
