import os
import tensorflow as tf
from model import create_model
from preprocess import create_generators
import argparse

def train(data_dir, epochs=10, batch_size=32, save_path='model.h5'):
    print(f"Loading data from {data_dir}...")
    train_gen, val_gen = create_generators(data_dir, batch_size=batch_size)
    
    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    print(f"Found {num_classes} classes: {class_indices}")
    
    # Save class indices
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    
    print("Building model...")
    model = create_model(num_classes=num_classes)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=3, 
        restore_best_weights=True
    )
    
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stopping]
    )
    
    print(f"Saving model to {save_path}...")
    model.save(save_path)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/food_mnist-master/images', help='Path to dataset images')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs, args.batch_size)
