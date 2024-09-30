from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
import pandas as pd
from PIL import Image
import urllib
import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

app = Flask(__name__)

model = tf.keras.models.load_model('sign.h5')

@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")

@app.route('/verify', methods=['POST'])
def verify_signature():
    if request.method == 'POST':
        image1 = request.files['signature1']
        image2 = request.files['signature2']
  
        img_path1 = "static/tests/" + image1.filename	
        img_path2 = "static/test/" + image2.filename	
        image1.save(img_path1)
        image2.save(img_path2)

    # Preprocess the images
        test_image1 = Image.open(image1)
        test_image1 = test_image1.resize((112, 112))
        test_image1 = img_to_array(test_image1)
        test_image1 = np.expand_dims(test_image1, axis=0)
        test_image1 = test_image1.astype('float32')

        test_image2 = Image.open(image2)
        test_image2 = test_image2.resize((112, 112))
        test_image2 = img_to_array(test_image2)
        test_image2 = np.expand_dims(test_image2, axis=0)
        test_image2 = test_image2.astype('float32')
    # print(img1)
	# print(img2)
    # Pass the images through the model to get the similarity score
        prediction = model.predict([test_image1, test_image2])
        unsimilarity_score = prediction[0][0]
        unsimilarity_percentage=unsimilarity_score*100
        print(f"Unsimiarity Score: {unsimilarity_percentage:.2f}%")

        if unsimilarity_percentage >= 50:
             result = 'The signature is Fake',unsimilarity_percentage
        else:
              result = 'The signature is Real',unsimilarity_percentage
  
            
    # Return the similarity score to the user
    return render_template('prediction.html', result=result, img_path1 = img_path1, img_path2 = img_path2)
    
@app.route("/chart")
def chart():
	return render_template('chart.html')   

if __name__ == '__main__':
    app.run(debug=True)