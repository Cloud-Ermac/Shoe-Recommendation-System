# @Author: Olayinka Hassan (Cloud_Ermac)
# import the necessary packages

from keras_preprocessing import image
from keras.layers import GlobalAvgPool2D
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# load the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalAvgPool2D()])

model.summary()


# function to get features of an image
def get_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    result = model.predict(img).flatten()
    # normalise the result
    return result / norm(result)


img_files = []
for shoe_images in os.listdir('Dataset/resized_images'):
    shoe_images_path = os.path.join('Dataset/resized_images', shoe_images)
    if os.path.isdir(shoe_images_path):
        for image_file in os.listdir(shoe_images_path):
            image_file_path = os.path.join(shoe_images_path, image_file)
            abs_image_file_path = os.path.abspath(image_file_path)
            img_files.append(abs_image_file_path)
pickle.dump(img_files, open('img_files2.pkl', 'wb'))

# extract features
features = []
for files in tqdm(img_files):
    features_list = get_features(files, model)
    features.append(features_list)
pickle.dump(features, open('img_features.pkl', 'wb'))
