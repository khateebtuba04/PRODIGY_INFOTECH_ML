import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
from model import create_model

# Configuration
DATA_DIR = os.path.join('train', 'train') # Path to the inner train directory
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 20

def train():
    # Data Augmentation and Normalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False, # Hand gestures might be sensitive to flip (left vs right hand)
        validation_split=0.2
    )

    print(f"Loading data from {os.path.abspath(DATA_DIR)}")

    # Load Training Data
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Load Validation Data
    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Create Model
    model = create_model(input_shape=(128, 128, 1), num_classes=NUM_CLASSES)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # Train Model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save Model
    model.save('hand_gesture_model.h5')
    print("Model saved to hand_gesture_model.h5")

    # Save Class Indices mapping
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    print("Class indices saved to class_indices.json")

if __name__ == '__main__':
    train()
