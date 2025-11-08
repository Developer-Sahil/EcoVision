import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# OPTIMIZED Configuration for FAST training
IMG_SIZE = 96  # Even smaller - still good for 2 classes
BATCH_SIZE = 128  # Larger batches = faster
EPOCHS = 5  # Just 5 epochs is enough for binary classification
DATASET_PATH = 'DATASET'

print("="*50)
print("FAST TRAINING MODE ENABLED")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("="*50)

# Minimal data augmentation for speed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'TRAIN'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'TEST'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Steps per epoch: {train_generator.samples // BATCH_SIZE}")

# SUPER SIMPLE AND FAST MODEL
def create_fast_model():
    model = keras.Sequential([
        # Just 2 conv layers - enough for binary classification
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Simple classifier
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Use MobileNetV2 for even better results (RECOMMENDED)
def create_mobilenet_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze - much faster!
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Choose model
print("\nChoose your model:")
print("1. Fast CNN (trains in ~1 minute)")
print("2. MobileNetV2 (better accuracy, trains in ~2 minutes)")

# Using MobileNetV2 by default - better accuracy for similar speed
model = create_mobilenet_model()  # Change to create_fast_model() if you prefer
print("✓ Using MobileNetV2 (Transfer Learning)")

# Compile with faster optimizer settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Calculate steps
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = test_generator.samples // BATCH_SIZE

print(f"\n{'='*50}")
print(f"Starting FAST training...")
print(f"Expected time: ~2-3 minutes total")
print(f"{'='*50}\n")

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=validation_steps,
    verbose=1
)

# Evaluate model
print("\n" + "="*50)
print("Evaluating model on test set...")
test_loss, test_acc = model.evaluate(test_generator, steps=validation_steps)
print(f"Test accuracy: {test_acc*100:.2f}%")
print("="*50)

# Save model
model.save('waste_classifier_model.h5')
print("\n✓ Model saved as 'waste_classifier_model.h5'")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✓ Training history saved as 'training_history.png'")

# Final summary
print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Model saved: waste_classifier_model.h5")
print("\nNext step: Run 'python app.py' to start the web app!")
print("="*50)