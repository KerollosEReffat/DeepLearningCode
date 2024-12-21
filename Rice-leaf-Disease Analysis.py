import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path to the directory containing the data
data_dir = '/kaggle/input/rice-leaf-disease-image'  

# Hyperparameters
img_size = (224, 224)

# The function loads and resizes images
def load_and_resize_image(file_path, target_shape=(224, 224)):
    image = cv2.imread(file_path)
    if image is not None:
        resized_image = cv2.resize(image, target_shape)
        return resized_image
    else:
        print(f"Unable to upload images from: {file_path}")
        return np.zeros((target_shape[0], target_shape[1], 3))

# The function loads images from a directory and assigns labels to them
def load_images_and_labels(img_size, directory, label_name):
    image_files = os.listdir(directory)
    images = []
    labels = []
    for file in image_files:
        if file.endswith('.jpg') or file.endswith('.JPG'):
            image_path = os.path.join(directory, file)
            image = load_and_resize_image(image_path, img_size)
            images.append(image)
            labels.append(label_name)  # Gán tên loại bệnh
    return np.array(images), np.array(labels)

# Load and label image layers
images_Bacterialblight, labels_Bacterialblight = load_images_and_labels(img_size, os.path.join(data_dir, 'Bacterialblight'), 'Bacterialblight')
images_Blast, labels_Blast = load_images_and_labels(img_size, os.path.join(data_dir, 'Blast'), 'Blast')
images_Brownspot, labels_Brownspot = load_images_and_labels(img_size, os.path.join(data_dir, 'Brownspot'), 'Brownspot')
images_Tungro, labels_Tungro = load_images_and_labels(img_size, os.path.join(data_dir, 'Tungro'), 'Tungro')

# Join data sets together
images = np.concatenate([images_Bacterialblight, images_Blast, images_Brownspot, images_Tungro])
labels = np.concatenate([labels_Bacterialblight, labels_Blast, labels_Brownspot, labels_Tungro])

# Custom Normalizing Images
images = images / 255.0
 
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNetV2, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Hyperparameters
# batch_size = 32
batch_size = 15
epochs = 15

layer_num_mobilenet = 128
layer_num_densenet = 64
layer_num_vgg = 256
layer_num_efficientnet = 256

learning_rate_mobilenet = 0.00005
learning_rate_densenet = 0.00005
learning_rate_vgg = 0.00005
learning_rate_efficientnet = 0.00101

# Model Building Function
def build_sequential_model(base_model, layer_num, learning_rate, num_classes):
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(layer_num, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and Evaluate Function
def train_and_evaluate_single_model(model, title, case, train_gen_no_aug = [], train_gen_aug = [], val_gen = [], test_gen = []):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    if case == "Train_No_Aug" and train_gen_no_aug != [] and val_gen != []:
        print(f"\nTraining {title} without Augmentation...")
        history_no_aug = model.fit(
            train_gen_no_aug[0],train_gen_no_aug[1],
            validation_data=val_gen,
            batch_size = batch_size,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr]
        )
        
        acc = history_no_aug.history['accuracy']
        val_acc = history_no_aug.history['val_accuracy']
    
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title(f'{title} Accuracy')
        plt.legend()
        plt.show()
    
        results[f"{title}_{case}"] = f"{title} Training Accuracy Without Augmentation : {max(history_no_aug.history['accuracy']):.4f}\n{title} Validation Accuracy Without Augmentation : {max(history_no_aug.history['val_accuracy']):.4f}"
    
    elif case == "Eval_No_Aug" and test_gen != []:
        print(f"\nEvaluating {title} on Test Data Without Augmentation...")
        test_loss_no_aug, test_acc_no_aug = model.evaluate(test_gen[0], test_gen[1])
        results[f"{title}_{case}"] = f"{title} Test Accuracy Without Augmentation : {test_acc_no_aug:.4f}"
    
    elif case == "Train_Aug" and train_gen_aug != [] and val_gen != []:
        print(f"\nTraining {title} with Augmentation...")
        history_aug = model.fit(
            datagen.flow(
                train_gen_aug[0],
                train_gen_aug[1],
                batch_size = batch_size
            ),
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr]
        )
                
        acc = history_aug.history['accuracy']
        val_acc = history_aug.history['val_accuracy']
    
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title(f'{title} Accuracy')
        plt.legend()
        plt.show()
        
        results[f"{title}_{case}"] = f"{title} Training Accuracy With Augmentation : {max(history_aug.history['accuracy']):.4f}\n{title} Validation Accuracy With Augmentation : {max(history_aug.history['val_accuracy']):.4f}"
    
    elif case == "Eval_Aug" and test_gen != []:
        print(f"\nEvaluating {title} on Test Data With Augmentation...")
        test_loss_aug, test_acc_aug = model.evaluate(test_gen[0], test_gen[1])
        results[f"{title}_{case}"] = f"{title} Test Accuracy With Augmentation : {test_acc_aug:.4f}"

    elif case == "Test" and test_gen != []:
        print("\n")
        # Get Predictions
        y_pred_probs = model.predict(test_gen[0])
        y_pred = np.argmax(y_pred_probs, axis=1)
        class_names = ["Bacterialblight","Blast","Brownspot","Tungro"]
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(test_gen[1], y_pred, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(test_gen[1], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix - {title}")
        plt.show()

    elif (case == "Train_No_Aug" and train_gen_no_aug == [] or val_gen == []):
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Training set Without Augmentation is None or Validation set is None or Both")
        exit()
        
    elif case == "Eval_No_Aug" and test_gen == []:
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Test set is None")
        exit()
        
    elif case == "Train_Aug" and train_gen_aug == [] or val_gen == []:
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Training set With Augmentation is None or Validation set is None or Both")
        exit()
        
    elif case == "Eval_Aug" and test_gen == []:
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Test set is None")
        exit()

    elif case == "Test" and test_gen == []:
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Test set is None")
        exit()

# Models
models = {
    "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3)),
    "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3)),
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
}

# Train and Compare
classes_number = 4
results = {}
train_generator_no_aug, val_generator, test_generator = (X_train, y_train), (X_val, y_val), (X_test, y_test)
for model_name, base_model in models.items():
    print(f"\nInitializing {model_name}...")

    if model_name == "MobileNetV2":
        model = build_sequential_model(base_model, layer_num_mobilenet, learning_rate_mobilenet, classes_number)

    elif model_name == "DenseNet121":
        model = build_sequential_model(base_model, layer_num_densenet, learning_rate_densenet, classes_number)

    elif model_name == "VGG16":
        model = build_sequential_model(base_model, layer_num_vgg, learning_rate_vgg, classes_number)
        
    elif model_name == "EfficientNetB0":
        model = build_sequential_model(base_model, layer_num_efficientnet, learning_rate_efficientnet, classes_number)

    # print("\n")
    # base_model.summary()

    train_and_evaluate_single_model(
        model,
        model_name,
        "Train_No_Aug",
        train_gen_no_aug = train_generator_no_aug,
        val_gen = val_generator
    )

    train_and_evaluate_single_model(
        model,
        model_name,
        "Eval_No_Aug",
        test_gen = test_generator
    )

    # train_and_evaluate_single_model(
    #     model,
    #     model_name,
    #     "Train_Aug",
    #     train_gen_aug = train_generator_aug,
    #     val_gen = val_generator
    # )

    # train_and_evaluate_single_model(
    #     model,
    #     model_name,
    #     "Eval_Aug",
    #     test_gen = test_generator
    # )

    train_and_evaluate_single_model(
        model,
        model_name,
        "Test",
        test_gen = test_generator
    )

print("================================================================================================================================")
# Results Comparison
print("Accuracy Comparison:")
for _, value in results.items():
    print(f"\n{value}")
print("================================================================================================================================")
