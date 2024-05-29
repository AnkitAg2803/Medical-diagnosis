import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import cv2
from sklearn.model_selection import train_test_split

# Function to load images from a folder
def load_images_from_folder(folder, img_size):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img)
                    labels.append(subdir)
                else:
                    print(f"Failed to load image: {img_path}")
    return images, labels

# Function to preprocess data
def preprocess_data(img_size, base_dir):
    images = []
    labels = []
    for dataset_type in ['train', 'test', 'val']:
        dataset_path = os.path.join(base_dir, dataset_type)
        imgs, lbls = load_images_from_folder(dataset_path, img_size)
        images.extend(imgs)
        labels.extend(lbls)
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Debug: Print dataset size
    print(f"Total images loaded: {len(images)}")
    print(f"Total labels loaded: {len(labels)}")
    
    # Check if data is loaded
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No images or labels found. Check the dataset path and structure.")

    # Normalize the images
    images = images / 255.0
    
    # Convert labels to numeric
    label_dict = {label: idx for idx, label in enumerate(np.unique(labels))}
    numeric_labels = np.array([label_dict[label] for label in labels])
    
    return train_test_split(images, numeric_labels, test_size=0.2, random_state=42)

# Prepare the Data
img_size = 128
base_dir = r'F:\code clause\chest_xray'
X_train, X_test, y_train, y_test = preprocess_data(img_size, base_dir)

# Build the Model
def build_model(img_size, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(os.listdir(os.path.join(base_dir, 'train')))
model = build_model(img_size, num_classes)

# Train the Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Make Predictions
def predict_image(model, img_path, img_size):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image. Check if the path is correct: {img_path}")
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    return predictions

# Example usage
img_path = 'path/to/test/image.jpg'  # Update this to an actual image path
try:
    predictions = predict_image(model, img_path, img_size)
    predicted_class = np.argmax(predictions)
    print(f'Predicted class: {predicted_class}')
except ValueError as e:
    print(e)

# Save the model
model.save('medical_image_model.h5')
