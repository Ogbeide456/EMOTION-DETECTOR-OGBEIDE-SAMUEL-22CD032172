import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size and batch setup
img_size = (48, 48)
batch_size = 64

# Load dataset (use your folder path if downloaded locally)
train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/train', target_size=img_size, color_mode='grayscale', class_mode='categorical')

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/test', target_size=img_size, color_mode='grayscale', class_mode='categorical')

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, epochs=10, validation_data=test_data)

# Save model
model.save('face_emotionModel.h5')
print("Model saved successfully!")
