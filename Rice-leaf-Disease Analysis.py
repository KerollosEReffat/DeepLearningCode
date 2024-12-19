import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNetV2, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Dataset Path
data_dir = "/kaggle/input/rice-leaf-disease-image"

# Hyperparameters
img_size = (224, 224)
batch_size = 32
epochs = 15

layer_num_mobilenet = 128
layer_num_densenet = 64
layer_num_vgg = 256
layer_num_efficientnet = 256

learning_rate_mobilenet = 0.00007
learning_rate_densenet = 0.00005
learning_rate_vgg = 0.000007
learning_rate_efficientnet = 0.001111

# Function to Create Data Generators
def create_data_generators(data_dir, img_size, batch_size, augment=False):
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2
        )
    
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

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
            train_gen_no_aug,
            validation_data=val_gen,
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
    
        results[f"{title}_{case}"] = f"{title} Validation Accuracy Without Augmentation : {max(history_no_aug.history['val_accuracy']):.4f}"
    
    elif case == "Eval_No_Aug" and test_gen != []:
        print(f"\nEvaluating {title} on Test Data Without Augmentation...")
        test_loss_no_aug, test_acc_no_aug = model.evaluate(test_gen)
        results[f"{title}_{case}"] = f"{title} Test Accuracy Without Augmentation : {test_acc_no_aug:.4f}"
    
    elif case == "Train_Aug" and train_gen_aug != [] and val_gen != []:
        print(f"\nTraining {title} with Augmentation...")
        history_aug = model.fit(
            train_gen_aug,
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
        
        results[f"{title}_{case}"] = f"{title} Validation Accuracy With Augmentation : {max(history_aug.history['val_accuracy']):.4f}"
    
    elif case == "Eval_Aug" and test_gen != []:
        print(f"\nEvaluating {title} on Test Data With Augmentation...")
        test_loss_aug, test_acc_aug = model.evaluate(test_gen)
        results[f"{title}_{case}"] = f"{title} Test Accuracy With Augmentation : {test_acc_aug:.4f}"

    elif (case == "Train_No_Aug" and train_gen_no_aug == [] or val_gen == []):
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Training set Without Augmentation is None , Validation set is None or Both")
        exit()
        
    elif case == "Eval_No_Aug" and test_gen == []:
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Test set Without Augmentation is None")
        exit()
        
    elif case == "Train_Aug" and train_gen_aug == [] or val_gen == []:
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Training set With Augmentation is None , Validation set is None or Both")
        exit()
        
    elif case == "Eval_Aug" and test_gen == []:
        print(f"Now, The Model Running is {title} , The Case Running is {case} and The Error is Test set With Augmentation is None")
        exit()

# Models
models = {
    "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3)),
    "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3)),
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
}

print("================================================================================================================================")
train_generator_no_aug, val_generator, _ = create_data_generators(data_dir, img_size, batch_size, augment=False)

train_no_aug_num = train_generator_no_aug.samples
vail_num = val_generator.samples

print(f"\nNumber of All Image: {train_no_aug_num + vail_num}\n")
print(f"Number of X Training Images : {train_no_aug_num}\n")
print(f"Number of X Validation and Test Images : {vail_num}\n")

# Shape of images
imgs_batch = next(train_generator_no_aug)
image_shape = imgs_batch[0].shape

print(f"Image size (height, width, channels): {image_shape}\n")
print("================================================================================================================================")

# Train and Compare
classes_number = train_generator_no_aug.num_classes
results = {}
for model_name, base_model in models.items():
    print(f"\nInitializing {model_name}...")
    train_generator_no_aug, val_generator, test_generator = create_data_generators(data_dir, img_size, batch_size, augment=False)
    # train_generator_aug, _, _ = create_data_generators(data_dir, img_size, batch_size, augment=True)

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

print("================================================================================================================================")
# Results Comparison
print("\nAccuracy Comparison:")
for case , value in results.items():
    print(f"\n{case} : {value}")
print("================================================================================================================================")
