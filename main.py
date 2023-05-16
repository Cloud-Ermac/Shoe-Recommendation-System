# @Author: Olayinka Hassan (Cloud_Ermac)


import os
import cv2
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import GlobalAvgPool2D
from keras.models import Sequential
from keras_preprocessing.image import load_img, img_to_array
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Setting the title for the application
st.title("Shoe Recommendation System")
# Change the working directory to where the file is located
os.chdir('/Users/saint-ermac/Documents/AInRB/RecSysApp')
# Load the trained  model for shoe recognition
model = load_model("shoe_recognition_model.h5")
# Load the trained model for shoe recommendation
model2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model2.trainable = False
model2 = Sequential([model2, GlobalAvgPool2D()])
# Load the recommendation data
features_list = pickle.load(open('img_features.pkl', 'rb'))
img_files = pickle.load(open('img_files2.pkl', 'rb'))


# Function to get features of an image
def get_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    result = model.predict(img).flatten()
    # normalise the result
    return result / norm(result)


# Function to get the indices of the top 6 nearest images
def get_recommendations(features, feature_list):
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    nbrs.fit(feature_list)
    distances, indices = nbrs.kneighbors([features])
    return indices


# Function to save the uploaded file to a directory
def sv_file(file):
    try:
        # Create a directory named 'uploads' if it doesn't exist
        if not os.path.exists('Dataset/uploads'):
            os.makedirs('Dataset/uploads')

        # Save the uploaded file in the 'uploads' directory with the original file name
        with open(os.path.join("Dataset/uploads", file.name), "wb") as f:
            f.write(file.getbuffer())

        # If the file is saved successfully, return True
        return True

    except:
        # If there's any exception while saving the file, return False
        return False


# predict shoe type
def predict_shoe_type(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Mapping the predicted class to the class name
    class_names = {"Boots": 0, "Sandals": 1, "Shoes": 2, "Slippers": 3, "Ankle": 4, "Knee High": 5,
                   "Mid-Calf": 6, "Over the Knee": 7, "Prewalker Boots": 8, "Athletic": 9, "Boat Shoes": 10,
                   "Clogs and Mules": 11,
                   "Crib Shoes": 12, "Firstwalker": 13, "Flats": 14, "Heels": 15, "Loafers": 16, "Oxfords": 17,
                   "Prewalker": 18,
                   "Sneakers and Athletic Shoes": 19, "Boot": 20, "Firstwalkers": 21, "Rain": 22, "Sneaker": 23,
                   "Cap Toe": 24,
                   "Closed Toe": 25, "Dorsay": 26, "Espadrille": 27, "Fisherman": 28, "Flip Flops": 29, "Gladiator": 30,
                   "Lace Up": 31, "Maryjane": 32, "Open Toe": 33, "Peep Toe": 34, "Slide": 35, "Sling Back": 36,
                   "Slipper Heels": 37, "Thong": 38, "Wedge": 39, "Wing Tips": 40}

    predicted_class_name = [k for k, v in class_names.items() if v == predicted_class][0]
    return predicted_class_name


# read and preprocess the image for prediction
def read_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Create a file uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    img = read_image(uploaded_image)
    size = (400, 400)
    resized_image = uploaded_image.resize(size)
    st.image(resized_image)

    if st.button("Identify Footwear"):
        predicted_class_name = predict_shoe_type(img)
        st.write(f"Footwear: {predicted_class_name}")

    if st.button("Recommend"):
        if sv_file(uploaded_file):
            features = get_features(os.path.join("Dataset/uploads", uploaded_file.name), model2)
            image_indices = get_recommendations(features, features_list)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.header('1')
                st.image(img_files[image_indices[0][0]])
            with col2:
                st.header('2')
                st.image(img_files[image_indices[0][1]])
            with col3:
                st.header('3')
                st.image(img_files[image_indices[0][2]])
            with col4:
                st.header('4')
                st.image(img_files[image_indices[0][3]])
            with col5:
                st.header('5')
                st.image(img_files[image_indices[0][4]])
        else:
            st.write('Error in uploading file')
