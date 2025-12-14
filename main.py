import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from pathlib import Path

# Function to load and encode background image
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load background image
bg_image = get_base64_image("newbg.jpg")
bg_style = ""
if bg_image:
    bg_style = f"""
    background-image: 
        linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)),
        url('data:image/jpeg;base64,{bg_image}');
    """
else:
    bg_style = """
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    """

# Enhanced Custom CSS
st.markdown(
    f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Main container with background image */
    .main {{
        {bg_style}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Alternative: Use local image if available */
    .stApp {{
        {bg_style}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
        box-shadow: 2px 0 10px rgba(0,0,0,0.5);
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    
    /* Title styling */
    .main-title {{
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a8d5ff 0%, #e0c3fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        filter: drop-shadow(0 0 20px rgba(168, 213, 255, 0.5));
    }}
    
    .sub-title {{
        text-align: center;
        font-size: 1.2rem;
        color: #e8e8e8;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }}
    
    /* Card styling */
    .info-card {{
        background: rgba(20, 20, 35, 0.85);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.6);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }}
    
    .info-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);
        border-left-color: #a8d5ff;
    }}
    
    .info-card h2, .info-card h3 {{
        color: #a8d5ff !important;
    }}
    
    .info-card p, .info-card li {{
        color: #e0e0e0 !important;
    }}
    
    /* Feature box */
    .feature-box {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }}
    
    .feature-box h3 {{
        color: white;
        margin-bottom: 0.5rem;
    }}
    
    /* Button styling */
    .stButton>button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1.1rem;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }}
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {{
        background: rgba(20, 20, 35, 0.85);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }}
    
    [data-testid="stFileUploader"] label {{
        color: #e0e0e0 !important;
    }}
    
    /* Success message styling */
    .stSuccess {{
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }}
    
    /* Info message styling */
    .stInfo {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }}
    
    /* Image container */
    .image-container {{
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }}
    
    /* Disease info box */
    .disease-info {{
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    
    /* Sidebar title */
    [data-testid="stSidebar"] h1 {{
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 2rem;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Step indicator */
    .step-indicator {{
        background: rgba(20, 20, 35, 0.85);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }}
    
    .step-indicator strong {{
        color: #a8d5ff;
    }}
    
    .step-indicator p {{
        color: #c0c0c0 !important;
    }}
    
    .step-number {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 0.5rem;
    }}
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
st.sidebar.markdown("# 🌿 Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("📍 Navigate To", ["Home", "About", "Disease Recognition"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 2rem;'>
        <p style='color: white; margin: 0;'>🔬 AI-Powered Detection</p>
        <p style='color: white; margin: 0; font-size: 0.9rem;'>Protecting Your Crops</p>
    </div>
""", unsafe_allow_html=True)

#Main Page
if app_mode == "Home":
    st.markdown("<h1 class='main-title'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>AI-Powered Agricultural Solution for Healthier Crops</p>", unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>👋 Welcome to the Future of Agriculture!</h2>
            <p style='font-size: 1.1rem; line-height: 1.8; color: #555;'>
                Our mission is to help farmers and agricultural enthusiasts identify plant diseases efficiently using 
                cutting-edge artificial intelligence. Upload an image of a plant leaf, and our advanced deep learning 
                model will analyze it to detect any signs of diseases in seconds!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("<h2 style='text-align: center; color: #667eea; margin: 2rem 0;'>🔍 How It Works</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='step-indicator'>
                <span class='step-number'>1</span>
                <strong>Upload Image</strong>
                <p style='margin-top: 0.5rem; color: #666;'>Navigate to Disease Recognition and upload a clear image of the plant leaf</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='step-indicator'>
                <span class='step-number'>2</span>
                <strong>AI Analysis</strong>
                <p style='margin-top: 0.5rem; color: #666;'>Our CNN model processes the image using advanced algorithms</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='step-indicator'>
                <span class='step-number'>3</span>
                <strong>Get Results</strong>
                <p style='margin-top: 0.5rem; color: #666;'>Receive instant diagnosis with detailed information and recommendations</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("<h2 style='text-align: center; color: #667eea; margin: 3rem 0 2rem 0;'>✨ Why Choose Us?</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='feature-box'>
                <h3>🎯 High Accuracy</h3>
                <p>97%+ accuracy using state-of-the-art CNN architecture trained on 87,000+ images</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='feature-box'>
                <h3>⚡ Lightning Fast</h3>
                <p>Get results in seconds with our optimized deep learning model</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='feature-box'>
                <h3>🌍 38 Disease Classes</h3>
                <p>Detect diseases across multiple plant species including tomato, potato, corn, and more</p>
            </div>
        """, unsafe_allow_html=True)
    
    # CTA section
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class='info-card' style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;'>
                <h2 style='color: white;'>🚀 Ready to Get Started?</h2>
                <p style='color: white; font-size: 1.1rem;'>Click on <strong>Disease Recognition</strong> in the sidebar to upload an image and experience the power of AI!</p>
            </div>
        """, unsafe_allow_html=True)

#About Project
elif app_mode == "About":
    st.markdown("<h1 class='main-title'>📚 About This Project</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Understanding the Technology Behind Plant Disease Detection</p>", unsafe_allow_html=True)
    
    # Project overview
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: #667eea;'>🔬 Project Overview</h2>
            <p style='font-size: 1.1rem; line-height: 1.8; color: #555;'>
                This project leverages the power of Convolutional Neural Networks (CNN) and TensorFlow to identify 
                plant diseases from leaf images. Our model has been trained on a comprehensive dataset to provide 
                accurate and reliable disease detection across multiple plant species.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-card'>
                <h3 style='color: #667eea;'>📊 Dataset Information</h3>
                <p style='line-height: 1.8; color: #555;'>
                    Our model is trained on the <strong>New Plant Diseases Dataset (Augmented)</strong> 
                    from Kaggle, which has been enhanced using offline augmentation techniques for improved accuracy.
                </p>
                <ul style='line-height: 2; color: #555;'>
                    <li>📷 <strong>87,000+</strong> RGB images</li>
                    <li>🏷️ <strong>38</strong> different classes</li>
                    <li>🎯 <strong>80/20</strong> train/validation split</li>
                    <li>✅ Healthy and diseased samples</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <h3 style='color: #667eea;'>📁 Dataset Breakdown</h3>
                <div style='margin-top: 1rem;'>
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                        <strong>Training Set</strong><br>70,295 images
                    </div>
                    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                        <strong>Validation Set</strong><br>17,572 images
                    </div>
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                        <strong>Test Set</strong><br>33 images
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Model architecture
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: #667eea;'>🧠 Model Architecture</h2>
            <p style='font-size: 1.1rem; line-height: 1.8; color: #555;'>
                Our Convolutional Neural Network consists of multiple layers designed to extract and learn 
                complex patterns from plant leaf images:
            </p>
            <ul style='line-height: 2; color: #555;'>
                <li>🔷 <strong>5 Convolutional Blocks</strong> with increasing filter depths (32 → 512)</li>
                <li>🔹 <strong>MaxPooling Layers</strong> for spatial dimension reduction</li>
                <li>🔸 <strong>Dropout Layers</strong> to prevent overfitting</li>
                <li>🔶 <strong>Dense Layers</strong> for final classification</li>
                <li>✨ <strong>Softmax Activation</strong> for multi-class prediction</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: #a8d5ff;'>📈 Model Performance</h2>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;'>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <h3 style='color: #38ef7d; font-size: 2.5rem; margin: 0;'>97.5%</h3>
                    <p style='color: #c0c0c0; margin: 0;'>Training Accuracy</p>
                </div>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <h3 style='color: #38ef7d; font-size: 2.5rem; margin: 0;'>95.4%</h3>
                    <p style='color: #c0c0c0; margin: 0;'>Validation Accuracy</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Technologies used
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: #a8d5ff;'>💻 Technologies Used</h2>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;'>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <strong style='color: #a8d5ff;'>TensorFlow</strong><br>
                    <span style='color: #c0c0c0;'>Deep Learning</span>
                </div>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <strong style='color: #a8d5ff;'>Keras</strong><br>
                    <span style='color: #c0c0c0;'>Neural Networks</span>
                </div>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <strong style='color: #a8d5ff;'>Streamlit</strong><br>
                    <span style='color: #c0c0c0;'>Web Interface</span>
                </div>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <strong style='color: #a8d5ff;'>NumPy</strong><br>
                    <span style='color: #c0c0c0;'>Data Processing</span>
                </div>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <strong style='color: #a8d5ff;'>Matplotlib</strong><br>
                    <span style='color: #c0c0c0;'>Visualization</span>
                </div>
                <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.4);'>
                    <strong style='color: #a8d5ff;'>OpenCV</strong><br>
                    <span style='color: #c0c0c0;'>Image Processing</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

#Prediction Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 class='main-title'>🔬 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Upload a plant leaf image for instant AI-powered disease detection</p>", unsafe_allow_html=True)
    
    # Upload section
    st.markdown("""
        <div class='info-card'>
            <h3 style='color: #667eea;'>📤 Upload Plant Leaf Image</h3>
            <p style='color: #666;'>Please upload a clear image of the plant leaf. Supported formats: JPG, JPEG, PNG</p>
        </div>
    """, unsafe_allow_html=True)
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, caption='📷 Uploaded Image', use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔍 Analyze Image")
        
        if predict_button:
            with st.spinner('🧠 Analyzing image with AI model...'):
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
                
                # Format the prediction name
                display_name = predicted_class.replace('___', ' - ').replace('_', ' ')
                
                # Determine if healthy or diseased
                is_healthy = 'healthy' in predicted_class.lower()
                
                if is_healthy:
                    st.markdown(f"""
                        <div class='info-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; border: none; text-align: center;'>
                            <h2 style='color: white;'>✅ Great News!</h2>
                            <h3 style='color: white; font-size: 1.8rem;'>{display_name}</h3>
                            <p style='color: white; font-size: 1.1rem;'>The plant appears to be healthy!</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='info-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; text-align: center;'>
                            <h2 style='color: white;'>⚠️ Disease Detected</h2>
                            <h3 style='color: white; font-size: 1.8rem;'>{display_name}</h3>
                            <p style='color: white; font-size: 1.1rem;'>Please review the information below</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Disease information
                st.markdown("""
                    <div class='info-card'>
                        <h3 style='color: #667eea;'>📋 Detailed Information</h3>
                """, unsafe_allow_html=True)
                
                # Display additional information based on the predicted class
            if predicted_class == 'Apple___Apple_scab':
                st.write("**About:** Apple scab is caused by the fungus Venturia inaequalis, resulting in dark, velvety spots on leaves and fruit. Infected leaves may curl and fall prematurely, leading to reduced fruit yield and quality.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply preventative fungicide sprays (captan, myclobutanil) in early spring</li>
                            <li>Plant resistant apple varieties (Enterprise, Liberty, Pristine)</li>
                            <li>Remove and destroy fallen leaves in autumn</li>
                            <li>Prune trees to improve air circulation</li>
                            <li>Avoid overhead irrigation to reduce leaf wetness</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Apple___Black_rot':
                st.write("**About:** Black rot, caused by the fungus Botryosphaeria obtusa, manifests as concentric dark brown to black lesions on leaves and fruit. It can also cause limb cankers and fruit rot, significantly affecting tree health and productivity.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Prune and destroy infected limbs and mummified fruits</li>
                            <li>Apply fungicides (captan, thiophanate-methyl) during bloom</li>
                            <li>Remove dead wood and cankers during dormant season</li>
                            <li>Maintain proper tree nutrition and vigor</li>
                            <li>Keep orchard floor clean of fallen debris</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Apple___Cedar_apple_rust':
                st.write("**About:** Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae, producing bright orange-yellow spots on apple leaves. It requires both apple and cedar (juniper) trees to complete its life cycle.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Remove nearby cedar/juniper trees within 2-3 miles if possible</li>
                            <li>Plant rust-resistant apple varieties (Redfree, Freedom)</li>
                            <li>Apply fungicides (myclobutanil, triadimefon) from pink bud to petal fall</li>
                            <li>Remove galls from juniper trees in early spring</li>
                            <li>Monitor weather conditions - apply fungicides before rain</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Apple___healthy':
                st.write("**About:** A healthy apple leaf is typically vibrant green, free from spots, lesions, or discoloration, indicating good tree health.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Continue regular watering and balanced fertilization</li>
                            <li>Prune annually for optimal air circulation</li>
                            <li>Monitor regularly for early signs of pests/diseases</li>
                            <li>Apply dormant oil spray in late winter</li>
                            <li>Mulch around base to retain moisture</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Blueberry___healthy':
                st.write("**About:** Healthy blueberry leaves are dark green, firm, and free from spots or discoloration, indicating optimal plant health.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Maintain soil pH between 4.5-5.5</li>
                            <li>Apply acidifying fertilizers (ammonium sulfate)</li>
                            <li>Mulch with pine bark or sawdust</li>
                            <li>Prune old canes to encourage new growth</li>
                            <li>Provide consistent moisture, especially during fruiting</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Cherry_(including_sour)___Powdery_mildew':
                st.write("**About:** Powdery mildew, caused by the fungus Podosphaera clandestina, appears as a white, powdery coating on leaves, stems, and fruit. It thrives in dry, warm conditions.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply sulfur-based fungicides at first sign of infection</li>
                            <li>Prune trees to improve air circulation</li>
                            <li>Avoid excessive nitrogen fertilization</li>
                            <li>Remove and destroy infected plant parts</li>
                            <li>Apply neem oil as organic treatment option</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Cherry_(including_sour)___healthy':
                st.write("**About:** Healthy cherry leaves are glossy green and free from spots or deformities, indicating good tree health.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Water deeply but infrequently</li>
                            <li>Apply balanced fertilizer in early spring</li>
                            <li>Prune during dormant season</li>
                            <li>Protect from birds during fruiting</li>
                            <li>Monitor for pest infestations regularly</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
                st.write("**About:** Gray leaf spot, caused by the fungus Cercospora zeae-maydis, presents as rectangular gray or tan lesions on leaves. It thrives in warm, humid environments.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Plant resistant corn hybrids</li>
                            <li>Practice crop rotation (2-3 year cycle)</li>
                            <li>Till under crop residue after harvest</li>
                            <li>Apply foliar fungicides (strobilurins, triazoles)</li>
                            <li>Avoid planting in poorly drained fields</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Corn_(maize)___Common_rust_':
                st.write("**About:** Common rust, caused by the fungus Puccinia sorghi, forms reddish-brown pustules on both leaf surfaces. It can cause significant defoliation and yield reduction.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Plant rust-resistant corn varieties</li>
                            <li>Apply fungicides early when pustules first appear</li>
                            <li>Scout fields regularly during humid weather</li>
                            <li>Ensure adequate plant spacing</li>
                            <li>Avoid late planting which increases risk</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Corn_(maize)___Northern_Leaf_Blight':
                st.write("**About:** Northern leaf blight, caused by *Exserohilum turcicum*, manifests as long, gray-green lesions on leaves, which can merge and cause significant tissue death.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Select hybrids with Ht resistance genes</li>
                            <li>Rotate crops to reduce inoculum</li>
                            <li>Apply fungicides at tasseling if conditions favor disease</li>
                            <li>Destroy crop debris after harvest</li>
                            <li>Avoid continuous corn planting</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Corn_(maize)___healthy':
                st.write("**About:** Healthy corn leaves are vibrant green and free from lesions, spots, or discoloration, indicating optimal plant health.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Ensure adequate nitrogen fertilization</li>
                            <li>Maintain consistent soil moisture</li>
                            <li>Scout regularly for pests and diseases</li>
                            <li>Control weeds to reduce competition</li>
                            <li>Plan crop rotation for next season</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Grape___Black_rot':
                st.write("**About:** Black rot, caused by the fungus *Guignardia bidwellii*, results in black lesions on leaves, shoots, and fruit. It thrives in warm, humid conditions.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Remove mummified berries and infected canes</li>
                            <li>Apply fungicides (mancozeb, myclobutanil) from bud break</li>
                            <li>Maintain canopy management for air circulation</li>
                            <li>Prune properly to reduce humidity</li>
                            <li>Scout weekly during wet weather</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Grape___Esca_(Black_Measles)':
                st.write("**About:** Esca, also known as black measles, is a complex disease involving multiple fungi, causing dark streaks and spots on leaves and berries.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Remove and destroy severely infected vines</li>
                            <li>Avoid pruning during wet weather</li>
                            <li>Protect pruning wounds with wound sealant</li>
                            <li>Practice vineyard sanitation</li>
                            <li>Remedial surgery (trunk renewal) for valuable vines</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
                st.write("**About:** Leaf blight, caused by *Pseudocercospora vitis*, results in angular brown spots on leaves, leading to premature defoliation.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply copper-based fungicides preventatively</li>
                            <li>Remove infected leaves and debris</li>
                            <li>Improve air circulation through pruning</li>
                            <li>Avoid overhead irrigation</li>
                            <li>Apply fungicides before wet periods</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Grape___healthy':
                st.write("**About:** Healthy grape leaves are bright green and free from spots, lesions, or discoloration, supporting vigorous vine growth.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Maintain proper canopy management</li>
                            <li>Apply balanced fertilization</li>
                            <li>Ensure adequate but not excessive watering</li>
                            <li>Train vines for optimal sun exposure</li>
                            <li>Monitor for early signs of problems</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Orange___Haunglongbing_(Citrus_greening)':
                st.write("**About:** Huanglongbing (HLB), or citrus greening, is caused by the bacterium *Candidatus Liberibacter* spp., transmitted by the Asian citrus psyllid.")
                st.markdown("""
                    <div style='background: rgba(245, 87, 108, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #f5576c;'>
                        <strong style='color: #f5576c;'>⚠️ Critical Prevention (No Cure Available):</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Control Asian citrus psyllid populations aggressively</li>
                            <li>Remove and destroy infected trees immediately</li>
                            <li>Use certified disease-free nursery stock</li>
                            <li>Apply systemic insecticides (imidacloprid)</li>
                            <li>Implement area-wide psyllid management programs</li>
                            <li>Inspect trees regularly for symptoms</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Peach___Bacterial_spot':
                st.write("**About:** Bacterial spot, caused by *Xanthomonas arboricola pv. pruni*, results in dark, water-soaked lesions on leaves, fruit, and twigs.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Plant resistant varieties when available</li>
                            <li>Apply copper sprays during dormant season</li>
                            <li>Use oxytetracycline sprays during bloom</li>
                            <li>Avoid overhead irrigation</li>
                            <li>Remove infected plant material</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Peach___healthy':
                st.write("**About:** Healthy peach leaves are deep green and free from spots, lesions, or deformities, supporting robust growth.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Water deeply during dry periods</li>
                            <li>Apply balanced fertilizer in spring</li>
                            <li>Thin fruit for better quality</li>
                            <li>Prune annually for shape and air flow</li>
                            <li>Apply dormant spray in late winter</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Pepper,_bell___Bacterial_spot':
                st.write("**About:** Bacterial spot, caused by *Xanthomonas campestris pv. vesicatoria*, causes water-soaked spots on leaves, stems, and fruit.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Use certified disease-free seeds and transplants</li>
                            <li>Apply copper-based bactericides preventatively</li>
                            <li>Practice crop rotation (3+ years)</li>
                            <li>Avoid working with wet plants</li>
                            <li>Remove and destroy infected plants</li>
                            <li>Use drip irrigation instead of overhead</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Pepper,_bell___healthy':
                st.write("**About:** Healthy bell pepper leaves are glossy green and free from spots, lesions, or discoloration.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Provide consistent watering</li>
                            <li>Apply calcium to prevent blossom end rot</li>
                            <li>Stake plants for support</li>
                            <li>Mulch to retain moisture</li>
                            <li>Fertilize every 2-3 weeks during fruiting</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Potato___Early_blight':
                st.write("**About:** Early blight, caused by *Alternaria solani*, results in concentric brown spots on leaves and stems, leading to defoliation.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Practice 2-3 year crop rotation</li>
                            <li>Apply fungicides (chlorothalonil, mancozeb) preventatively</li>
                            <li>Remove infected plant debris</li>
                            <li>Use certified disease-free seed potatoes</li>
                            <li>Maintain adequate plant nutrition</li>
                            <li>Avoid overhead irrigation</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Potato___Late_blight':
                st.write("**About:** Late blight, caused by *Phytophthora infestans*, produces water-soaked lesions on leaves and stems, rapidly leading to plant collapse.")
                st.markdown("""
                    <div style='background: rgba(245, 87, 108, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #f5576c;'>
                        <strong style='color: #f5576c;'>⚠️ Urgent Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply fungicides immediately at first sign</li>
                            <li>Use resistant varieties (Defender, Elba)</li>
                            <li>Destroy all infected plant material</li>
                            <li>Ensure good drainage and air circulation</li>
                            <li>Monitor weather - apply protection before rain</li>
                            <li>Do not compost infected plants - burn or bury</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Potato___healthy':
                st.write("**About:** Healthy potato leaves are dark green and free from spots or lesions, supporting robust growth.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Hill soil around plants as they grow</li>
                            <li>Water consistently, 1-2 inches per week</li>
                            <li>Apply balanced fertilizer at planting</li>
                            <li>Scout regularly for pests</li>
                            <li>Plan rotation for next season</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Raspberry___healthy':
                st.write("**About:** Healthy raspberry leaves are vibrant green and free from spots, lesions, or deformities.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Prune out old canes after fruiting</li>
                            <li>Provide trellising for support</li>
                            <li>Mulch heavily to retain moisture</li>
                            <li>Apply balanced fertilizer in spring</li>
                            <li>Maintain good air circulation</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Soybean___healthy':
                st.write("**About:** Healthy soybean leaves are dark green and free from spots, lesions, or discoloration.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Monitor for pest pressure regularly</li>
                            <li>Ensure proper soil pH (6.0-7.0)</li>
                            <li>Inoculate seeds with rhizobium if new field</li>
                            <li>Control weeds early in season</li>
                            <li>Scout for disease symptoms weekly</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Squash___Powdery_mildew':
                st.write("**About:** Powdery mildew appears as a white, powdery coating on leaves, stems, and fruit. It thrives in dry, warm conditions.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply sulfur or potassium bicarbonate sprays</li>
                            <li>Use neem oil as organic treatment</li>
                            <li>Plant resistant varieties</li>
                            <li>Ensure adequate plant spacing</li>
                            <li>Water at base, not on foliage</li>
                            <li>Remove heavily infected leaves</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 'Strawberry___healthy':
                st.write("**About:** Healthy strawberry leaves are bright green and free from spots, lesions, or discoloration.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Renovate beds after harvest</li>
                            <li>Mulch with straw to prevent fruit rot</li>
                            <li>Remove runners to focus energy on fruit</li>
                            <li>Water consistently, avoid wetting foliage</li>
                            <li>Replace plants every 3-4 years</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Bacterial_spot':
                st.write("**About:** Bacterial spot results in small, water-soaked spots on leaves, stems, and fruit that can enlarge and become necrotic.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Use certified disease-free seeds</li>
                            <li>Apply copper-based sprays preventatively</li>
                            <li>Rotate crops for 2-3 years</li>
                            <li>Avoid overhead irrigation</li>
                            <li>Remove infected plant debris</li>
                            <li>Don't work with wet plants</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Early_blight':
                st.write("**About:** Early blight presents as concentric rings on older leaves, leading to defoliation and reduced fruit quality.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply fungicides (chlorothalonil) preventatively</li>
                            <li>Remove lower leaves touching soil</li>
                            <li>Mulch to prevent soil splash</li>
                            <li>Stake plants for air circulation</li>
                            <li>Practice crop rotation</li>
                            <li>Water at base of plants</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Late_blight':
                st.write("**About:** Late blight causes water-soaked lesions on leaves and stems, quickly leading to plant collapse and significant fruit rot.")
                st.markdown("""
                    <div style='background: rgba(245, 87, 108, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #f5576c;'>
                        <strong style='color: #f5576c;'>⚠️ Urgent Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply fungicides immediately upon detection</li>
                            <li>Remove and destroy ALL infected plants</li>
                            <li>Use resistant varieties (Defiant, Mountain Magic)</li>
                            <li>Monitor weather forecasts closely</li>
                            <li>Ensure excellent air circulation</li>
                            <li>Do NOT compost - destroy infected material</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Leaf_Mold':
                st.write("**About:** Leaf mold appears as yellow spots on the upper leaf surface and olive-green to gray mold on the underside.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Improve greenhouse ventilation</li>
                            <li>Reduce humidity below 85%</li>
                            <li>Space plants adequately</li>
                            <li>Apply fungicides if severe</li>
                            <li>Use resistant varieties</li>
                            <li>Remove infected leaves promptly</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Septoria_leaf_spot':
                st.write("**About:** Septoria leaf spot results in small, water-soaked spots that develop into circular lesions with dark borders and light centers.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply fungicides at first sign of disease</li>
                            <li>Remove infected lower leaves</li>
                            <li>Avoid overhead watering</li>
                            <li>Mulch to prevent soil splash</li>
                            <li>Rotate crops for 1-2 years</li>
                            <li>Destroy plant debris at season end</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Spider_mites Two-spotted_spider_mite':
                st.write("**About:** Two-spotted spider mites cause stippling and yellowing of leaves, leading to leaf drop and reduced plant vigor.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Spray plants with strong water jet</li>
                            <li>Apply insecticidal soap or neem oil</li>
                            <li>Release predatory mites (Phytoseiulus)</li>
                            <li>Maintain adequate humidity</li>
                            <li>Avoid dusty conditions</li>
                            <li>Use miticides for severe infestations</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Target_Spot':
                st.write("**About:** Target spot presents as dark, concentric lesions on leaves, stems, and fruit, leading to defoliation and fruit rot.")
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>🛡️ Prevention & Treatment:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Apply fungicides preventatively</li>
                            <li>Improve air circulation</li>
                            <li>Remove infected leaves</li>
                            <li>Practice crop rotation</li>
                            <li>Avoid excessive nitrogen</li>
                            <li>Water at plant base</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
                st.write("**About:** TYLCV is transmitted by whiteflies, causing yellowing and curling of leaves, stunted growth, and reduced fruit production.")
                st.markdown("""
                    <div style='background: rgba(245, 87, 108, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #f5576c;'>
                        <strong style='color: #f5576c;'>⚠️ Prevention (No Cure Available):</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Use TYLCV-resistant varieties</li>
                            <li>Control whitefly populations aggressively</li>
                            <li>Use reflective mulches to repel whiteflies</li>
                            <li>Install insect-proof netting</li>
                            <li>Remove and destroy infected plants</li>
                            <li>Apply insecticides for whitefly control</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___Tomato_mosaic_virus':
                st.write("**About:** Tomato mosaic virus causes mottled, discolored leaves, stunted growth, and reduced yields. It spreads through contaminated tools and hands.")
                st.markdown("""
                    <div style='background: rgba(245, 87, 108, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #f5576c;'>
                        <strong style='color: #f5576c;'>⚠️ Prevention (No Cure Available):</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Use resistant tomato varieties</li>
                            <li>Sanitize tools with 10% bleach solution</li>
                            <li>Wash hands before handling plants</li>
                            <li>Avoid tobacco use near plants</li>
                            <li>Remove and destroy infected plants</li>
                            <li>Use certified virus-free seeds</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            elif predicted_class == 'Tomato___healthy':
                st.write("**About:** Healthy tomato leaves are vibrant green and free from spots, lesions, or discoloration.")
                st.markdown("""
                    <div style='background: rgba(17, 153, 142, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #38ef7d;'>
                        <strong style='color: #38ef7d;'>✅ Maintenance Tips:</strong>
                        <ul style='color: #c0c0c0; margin-top: 0.5rem;'>
                            <li>Water consistently at plant base</li>
                            <li>Stake or cage for support</li>
                            <li>Prune suckers for indeterminate varieties</li>
                            <li>Fertilize every 2-3 weeks</li>
                            <li>Mulch to retain moisture and prevent disease</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.write("Additional information for this class is not available at the moment.")
    else:
        st.info("Please upload an image.")
