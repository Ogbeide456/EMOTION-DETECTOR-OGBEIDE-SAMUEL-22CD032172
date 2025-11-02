from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import sqlite3

app = Flask(__name__)
model = load_model('face_emotionModel.h5')

# Emotion labels (FER2013 standard)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create a simple database
def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, emotion TEXT)')
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filepath = 'static/' + file.filename
        file.save(filepath)

        # Load and preprocess image
        img = image.load_img(filepath, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        emotion = emotions[np.argmax(prediction)]

        # Save to database
        conn = sqlite3.connect('database.db')
        conn.execute("INSERT INTO predictions (emotion) VALUES (?)", (emotion,))
        conn.commit()
        conn.close()

        return render_template('index.html', emotion=emotion, image=file.filename)
    return render_template('index.html', emotion=None, image=None)

if __name__ == '__main__':
    init_db()
    if not os.path.exists('static'):
        os.mkdir('static')
    app.run(debug=True)
