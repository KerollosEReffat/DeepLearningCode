import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path to the directory containing the data
data_train_dir = '/kaggle/input/paddy-doctor/paddy-disease-classification/train_images'
data_test_dir = '/kaggle/input/paddy-doctor/paddy-disease-classification/test_images'

# Hyperparameters
img_size = (224, 224)

# The function loads and resizes images
def load_and_resize_image(file_path, target_shape=(224, 224)):
    image = cv2.imread(file_path)
    if image is not None:
        resized_image = cv2.resize(image, target_shape)
        return resized_image
    else:
        print(f"Unable to upload image from: {file_path}")
        return np.zeros((target_shape[0], target_shape[1], 3))

# The function loads images from a directory and assigns labels to them
def load_images_and_labels(img_size, directory, label_name = ""):
    image_files = os.listdir(directory)
    images = []
    labels = []
    for file in image_files:
        if file.endswith('.jpg') or file.endswith('.JPG'):
            image_path = os.path.join(directory, file)
            image = load_and_resize_image(image_path, img_size)
            images.append(image)
            labels.append(label_name)  # Gán tên loại bệnh
    if (label_name == ""):
        return np.array(images)
    else:
        return np.array(images), np.array(labels)
    

# Load and label image layers
images_Brown_Spot, labels_Brown_Spot = load_images_and_labels(img_size, os.path.join(data_train_dir, 'brown_spot'), 'Brown_Spot')
images_Dead_Heart, labels_Dead_Heart = load_images_and_labels(img_size, os.path.join(data_train_dir, 'dead_heart'), 'Dead_Heart')
images_Downy_Mildew, labels_Downy_Mildew = load_images_and_labels(img_size, os.path.join(data_train_dir, 'downy_mildew'), 'Downy_Mildew')
images_Hispa, labels_Hispa = load_images_and_labels(img_size, os.path.join(data_train_dir, 'hispa'), 'Hispa')
images_Tungro, labels_Tungro = load_images_and_labels(img_size, os.path.join(data_train_dir, 'tungro'), 'Tungro')
images_Normal, labels_Normal = load_images_and_labels(img_size, os.path.join(data_train_dir, 'normal'), 'Normal')

images_Test = load_images_and_labels(img_size, data_test_dir)

# Join data sets together
images = np.concatenate([
    images_Brown_Spot,
    images_Dead_Heart,
    images_Downy_Mildew,
    images_Hispa,
    images_Tungro,
    images_Normal
])
labels = np.concatenate([
    labels_Brown_Spot,
    labels_Dead_Heart,
    labels_Downy_Mildew,
    labels_Hispa,
    labels_Tungro,
    labels_Normal
])

# images = images / 255.0

# Convert labels to numbers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Divide the data into training set, validation set, and test set
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=(0.1/(0.1+0.2)), random_state=42)

print("Image size (height, width, channels):", images_Tungro[0].shape)
print("\n")
print("Number of All Image: ",len(labels_encoded))
print("\n")
print("Number of Y Train Images: ",len(y_train))
print("\n")
print("Number of Y Validation Images: ",len(y_val))
print("\n")
print("Number of Y Test Images: ",len(y_test))

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Define and compile the model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

efficientNetB0_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(6, activation='softmax')  
])

# Print the model summary
# base_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
# learning_rate = 0.0005
learning_rate = 0.001
epochs = 25
# batch_size = 19
batch_size = 32

efficientNetB0_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = efficientNetB0_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size = batch_size,
    epochs = epochs,
    callbacks=[early_stopping, reduce_lr]
)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('EfficientNetB0 Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = efficientNetB0_model.evaluate(X_test, y_test)
print("Accuracy of EfficientNet before Data Augmentation on the test set:", test_accuracy)
print("Loss of EfficientNetB0 before Data Augmentation on the test set:", test_loss)

# Get Predictions
y_pred_probs = efficientNetB0_model.predict(X_test)  # Predict probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class predictions
class_names = ["Brown_Spot", "Dead_Heart", "Downy_Mildew", "Hispa", "Tungro", "Normal"]

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)  # Compare ground truth (y_test) and predictions (y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - EfficientNetB0")
plt.show()
