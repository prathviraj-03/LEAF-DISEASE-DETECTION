import streamlit as st
import tensorflow as tf
import numpy as np

# Custom CSS for improved aesthetics
st.markdown(
    """
    <style>
    .title-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
        .title-text {
        text-align: center;
        font-size: 36px; 
        color: #ffffff;
    }        
    .header-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sidebar {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .content {
        margin-top: 20px;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
    }
    .button {
        background-color: #008CBA;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        margin-top: 10px;
        cursor: pointer;
       }
    </style>
    """,
    unsafe_allow_html=True
)

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if app_mode == "Home":
    st.title("LEAF DISEASE DETECTION")
    image_path = "BBG.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍

        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
        """)

#About Project
elif app_mode == "About":
    st.title("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
        This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purpose.
        #### Content
        1. train (70295 images)
        2. test (33 images)
        3. validation (17572 images)

        """)

#Prediction Page
elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        #Predict button
        if st.button("Predict"):
            st.success("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            predicted_class = class_name[result_index] if result_index < len(class_name) else "Unknown"
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
            # Display additional information based on the predicted class
            if predicted_class == 'Apple___Apple_scab':
                st.write(" Apple scab is caused by the fungus Venturia inaequalis, resulting in dark, velvety spots on leaves and fruit. Infected leaves may curl and fall prematurely, leading to reduced fruit yield and quality. Cool, wet conditions in spring and early summer favor the development of this disease. Preventative fungicide sprays and resistant apple varieties are key management strategies.")               
            elif predicted_class == 'Apple___Black_rot':
                st.write("Black rot, caused by the fungus Botryosphaeria obtusa, manifests as concentric dark brown to black lesions on leaves and fruit. It can also cause limb cankers and fruit rot, significantly affecting tree health and productivity. Warm, humid weather promotes the spread of black rot. Pruning infected limbs and applying fungicides can help control the disease.")
            # Add more elif statements for other classes...
            elif predicted_class == 'Apple___Cedar_apple_rust':
                st.write("Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae, producing bright orange-yellow spots on apple leaves. It requires both apple and cedar (juniper) trees to complete its life cycle, alternating between hosts. Infected leaves may drop prematurely, weakening the tree. Management includes removing nearby junipers and using fungicides.")
            elif predicted_class == 'Apple___healthy':
                st.write(" healthy apple leaf is typically vibrant green, free from spots, lesions, or discoloration, indicating good tree health. Proper care includes regular watering, balanced fertilization, and appropriate pruning. Monitoring for pests and diseases ensures early detection and treatment. Healthy leaves contribute to optimal photosynthesis and fruit production.")
            elif predicted_class == 'Blueberry___healthy':
                st.write(" Healthy blueberry leaves are dark green, firm, and free from spots or discoloration. Regular watering, mulching, and appropriate fertilization are essential for maintaining plant health. Proper pruning helps improve air circulation and sunlight penetration. Monitoring for pests and diseases ensures timely intervention and sustained plant vigor.")
            elif predicted_class == 'Cherry_(including_sour)___Powdery_mildew':
                st.write("Powdery mildew, caused by the fungus Podosphaera clandestina, appears as a white, powdery coating on leaves, stems, and fruit. It thrives in dry, warm conditions and can stunt growth and reduce fruit quality. Fungicide applications and pruning for better air circulation help manage the disease.")
            elif predicted_class == 'Cherry_(including_sour)___healthy':
                st.write("Healthy cherry leaves are glossy green and free from spots or deformities. Proper care includes adequate watering, balanced fertilization, and timely pruning. Regular monitoring for signs of pests and diseases is crucial. Healthy leaves contribute to robust growth and high-quality fruit production.")
            elif predicted_class == 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
                st.write(" Gray leaf spot, caused by the fungus Cercospora zeae-maydis, presents as rectangular gray or tan lesions on leaves. It thrives in warm, humid environments, potentially leading to significant yield loss. Crop rotation, resistant hybrids, and timely fungicide applications are effective control measures.")
            elif predicted_class == 'Corn_(maize)___Common_rust_':
                st.write("Common rust, caused by the fungus Puccinia sorghi, forms reddish-brown pustules on both leaf surfaces. It can cause significant defoliation and yield reduction under favorable conditions. Resistant varieties and fungicide applications help manage the disease.")
            elif predicted_class == 'Corn_(maize)___Northern_Leaf_Blight':
                st.write("Northern leaf blight, caused by *Exserohilum turcicum*, manifests as long, gray-green lesions on leaves, which can merge and cause significant tissue death. Cool, moist conditions favor its spread. Management includes planting resistant hybrids and applying fungicides when necessary.")
            elif predicted_class == 'Corn_(maize)___healthy':
                st.write("Healthy corn leaves are vibrant green and free from lesions, spots, or discoloration. Proper irrigation, balanced fertilization, and pest monitoring are vital for maintaining plant health. Healthy leaves ensure efficient photosynthesis, leading to robust plant growth and optimal yields.")
            elif predicted_class == 'Grape___Black_rot':
                st.write("Black rot, caused by the fungus *Guignardia bidwellii*, results in black lesions on leaves, shoots, and fruit. It thrives in warm, humid conditions and can devastate grape yields. Pruning infected parts and applying fungicides are essential for control.")

            elif predicted_class == 'Grape___Esca_(Black_Measles)':
                st.write("Esca, also known as black measles, is a complex disease involving multiple fungi, causing dark streaks and spots on leaves and berries. It can lead to vine decline and death. Management includes removing infected vines and avoiding wounding.")

            elif predicted_class == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
                st.write("Leaf blight, caused by *Pseudocercospora vitis*, results in angular brown spots on leaves, leading to premature defoliation. Warm, wet conditions favor its spread. Fungicides and good vineyard sanitation help manage the disease.")

            elif predicted_class == 'Grape___healthy':
                st.write("Healthy grape leaves are bright green and free from spots, lesions, or discoloration. Proper vineyard management, including adequate irrigation, fertilization, and pest control, is crucial. Healthy leaves support vigorous vine growth and high-quality grape production.")

            elif predicted_class == 'Orange___Haunglongbing_(Citrus_greening)':
                st.write("Huanglongbing (HLB), or citrus greening, is caused by the bacterium *Candidatus Liberibacter* spp., transmitted by the Asian citrus psyllid. Symptoms include yellowing leaves, misshapen fruit, and tree decline. There is no cure, so management focuses on controlling psyllid populations and removing infected trees.")

            elif predicted_class == 'Peach___Bacterial_spot':
                st.write("Bacterial spot, caused by *Xanthomonas arboricola pv. pruni*, results in dark, water-soaked lesions on leaves, fruit, and twigs. Wet, warm conditions promote its spread, leading to defoliation and fruit blemishes. Copper-based sprays and resistant varieties help manage the disease.")

            elif predicted_class == 'Peach___healthy':
                st.write("Healthy peach leaves are deep green and free from spots, lesions, or deformities. Regular watering, balanced fertilization, and proper pruning are essential for maintaining health. Monitoring for pests and diseases ensures early detection and intervention. Healthy leaves support robust growth and high-quality fruit production.")

            elif predicted_class == 'Pepper,_bell___Bacterial_spot':
                st.write("Bacterial spot, caused by *Xanthomonas campestris pv. vesicatoria*, causes water-soaked spots on leaves, stems, and fruit, leading to defoliation and fruit blemishes. Warm, wet conditions favor its spread. Copper-based fungicides and resistant varieties are effective management strategies.")

            elif predicted_class == 'Pepper,_bell___healthy':
                st.write("Healthy bell pepper leaves are glossy green and free from spots, lesions, or discoloration. Adequate watering, balanced fertilization, and regular pest monitoring are crucial for plant health. Healthy leaves contribute to vigorous growth and high-quality fruit production.")

            elif predicted_class == 'Potato___Early_blight':
                st.write("Early blight, caused by *Alternaria solani*, results in concentric brown spots on leaves and stems, leading to defoliation and reduced tuber quality. Warm, humid conditions favor its spread. Crop rotation, resistant varieties, and fungicide applications help manage the disease.")

            elif predicted_class == 'Potato___Late_blight':
                st.write("Late blight, caused by *Phytophthora infestans*, produces water-soaked lesions on leaves and stems, rapidly leading to plant collapse and tuber rot. Cool, wet conditions favor its spread. Management includes resistant varieties, proper field sanitation, and fungicide applications.")

            elif predicted_class == 'Potato___healthy':
                st.write("Healthy potato leaves are dark green and free from spots or lesions. Proper irrigation, balanced fertilization, and pest monitoring are essential for maintaining plant health. Healthy leaves support robust growth and high-quality tuber production.")

            elif predicted_class == 'Raspberry___healthy':
                st.write("Healthy raspberry leaves are vibrant green and free from spots, lesions, or deformities. Regular watering, balanced fertilization, and appropriate pruning are essential for maintaining plant health. Monitoring for pests and diseases ensures early detection and treatment. Healthy leaves contribute to vigorous growth and high-quality fruit production.")

            elif predicted_class == 'Soybean___healthy':
                st.write("Healthy soybean leaves are dark green and free from spots, lesions, or discoloration. Adequate irrigation, balanced fertilization, and regular pest monitoring are vital for maintaining plant health. Healthy leaves ensure efficient photosynthesis, leading to robust plant growth and optimal yields.")

            elif predicted_class == 'Squash___Powdery_mildew':
                st.write("Powdery mildew, caused by *Erysiphe cichoracearum* and *Podosphaera xanthii*, appears as a white, powdery coating on leaves, stems, and fruit. It thrives in dry, warm conditions and can stunt growth and reduce fruit quality. Fungicide applications and proper spacing for air circulation help manage the disease.")
            elif predicted_class == 'Strawberry___healthy':
                st.write("Healthy strawberry leaves are bright green and free from spots, lesions, or discoloration. Regular watering, balanced fertilization, and pest monitoring are crucial for plant health. Healthy leaves support vigorous growth and high-quality fruit production.")

            elif predicted_class == 'Tomato___Bacterial_spot':
                st.write("Bacterial spot, caused by *Xanthomonas campestris pv. vesicatoria*, results in small, water-soaked spots on leaves, stems, and fruit. The spots can enlarge, become necrotic, and merge, leading to significant damage and reduced yield. Warm, wet conditions favor its spread, and management includes copper-based sprays and resistant varieties.")

            elif predicted_class == 'Tomato___Early_blight':
                st.write("Early blight, caused by *Alternaria solani*, presents as concentric rings on older leaves, leading to defoliation and reduced fruit quality. It thrives in warm, humid conditions. Management includes crop rotation, resistant varieties, and timely fungicide applications.")

            elif predicted_class == 'Tomato___Late_blight':
                st.write("Late blight, caused by *Phytophthora infestans*, causes water-soaked lesions on leaves and stems, quickly leading to plant collapse and significant fruit rot. Cool, wet conditions favor its spread. Management strategies include using resistant varieties, ensuring proper field sanitation, and applying fungicides.")

            elif predicted_class == 'Tomato___Leaf_Mold':
                st.write("Leaf mold, caused by *Passalora fulva*, appears as yellow spots on the upper leaf surface and olive-green to gray mold on the underside. High humidity and poor ventilation favor its development. Managing the disease involves ensuring good air circulation, reducing humidity, and applying fungicides if necessary.")

            elif predicted_class == 'Tomato___Septoria_leaf_spot':
                st.write("Septoria leaf spot, caused by *Septoria lycopersici*, results in small, water-soaked spots that develop into circular lesions with dark borders and light centers. It can cause significant defoliation and reduced yields. Management includes crop rotation, removing infected plant debris, and applying fungicides.")

            elif predicted_class == 'Tomato___Spider_mites Two-spotted_spider_mite':
                st.write("Two-spotted spider mites (*Tetranychus urticae*) cause stippling and yellowing of leaves, leading to leaf drop and reduced plant vigor. They thrive in hot, dry conditions. Management includes using miticides, introducing natural predators, and maintaining adequate moisture levels.")

            elif predicted_class == 'Tomato___Target_Spot':
                st.write("Target spot, caused by *Corynespora cassiicola*, presents as dark, concentric lesions on leaves, stems, and fruit, leading to defoliation and fruit rot. Warm, humid conditions favor its spread. Effective management includes crop rotation, resistant varieties, and fungicide applications.")

            elif predicted_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
                st.write("Tomato yellow leaf curl virus (TYLCV) is transmitted by whiteflies, causing yellowing and curling of leaves, stunted growth, and reduced fruit production. Management focuses on controlling whitefly populations and using resistant tomato varieties.")

            elif predicted_class == 'Tomato___Tomato_mosaic_virus':
                st.write("Tomato mosaic virus (ToMV) causes mottled, discolored leaves, stunted growth, and reduced yields. It spreads through contaminated tools, hands, and plant debris. Preventative measures include using resistant varieties, sanitizing equipment, and removing infected plants.")

            elif predicted_class == 'Tomato___healthy':
                st.write("Healthy tomato leaves are vibrant green and free from spots, lesions, or discoloration. Proper watering, balanced fertilization, and pest monitoring are essential for plant health. Healthy leaves support robust growth and high-quality fruit production.")

            else:
                st.write("Additional information for this class is not available at the moment.")
    else:
        st.info("Please upload an image.")
