"""
MobileNetV2 Transfer Learning — Eye Model Comparison
=====================================================
This script trains TWO MobileNetV2 variants for eye open/close classification
and compares them against the custom CNN baseline.

Variant 1: FROZEN — MobileNetV2 base is completely frozen, only the top classifier trains.
Variant 2: FINE-TUNED — After initial frozen training, unfreeze the last 30 layers and train again with a low learning rate.

Results are printed at the end for easy comparison.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import time

# ===========================
# Dataset Setup (same as custom CNN)
# ===========================

TRAIN_DIR = r"D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED\TRAIN\EYE_TRAIN"
VAL_DIR   = r"D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED\VALIDATION\EYE_VAL"
TEST_DIR  = r"D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED\TEST\EYE_TEST"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(96,96), batch_size=32, class_mode='binary')
val_data   = val_datagen.flow_from_directory(VAL_DIR, target_size=(96,96), batch_size=32, class_mode='binary')
test_data  = test_datagen.flow_from_directory(TEST_DIR, target_size=(96,96), batch_size=32, class_mode='binary', shuffle=False)


# ===========================
# VARIANT 1: MobileNetV2 FROZEN
# ===========================
print("\n" + "="*60)
print("TRAINING VARIANT 1: MobileNetV2 FROZEN (feature extraction)")
print("="*60 + "\n")

# Load MobileNetV2 base without top classification layers
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze ALL layers — only our custom head trains

frozen_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

frozen_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

frozen_model.summary()

frozen_callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("Models/eye_model_mobilenet_frozen.h5", save_best_only=True)
]

start_time = time.time()
frozen_history = frozen_model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=frozen_callbacks
)
frozen_train_time = time.time() - start_time

# Evaluate frozen model
frozen_loss, frozen_acc = frozen_model.evaluate(test_data)


# ===========================
# VARIANT 2: MobileNetV2 FINE-TUNED
# ===========================
print("\n" + "="*60)
print("TRAINING VARIANT 2: MobileNetV2 FINE-TUNED (unfreeze last 30 layers)")
print("="*60 + "\n")

# Start from the frozen model and unfreeze the last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False  # Keep early layers frozen, only fine-tune last 30

# Re-compile with a LOWER learning rate (important to avoid destroying pre-trained weights)
frozen_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tuned_callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("Models/eye_model_mobilenet_tuned.h5", save_best_only=True)
]

start_time = time.time()
tuned_history = frozen_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=tuned_callbacks
)
tuned_train_time = time.time() - start_time

# Evaluate fine-tuned model
tuned_loss, tuned_acc = frozen_model.evaluate(test_data)


# ===========================
# LOAD & EVALUATE CUSTOM CNN BASELINE
# ===========================
print("\n" + "="*60)
print("EVALUATING BASELINE: Custom CNN")
print("="*60 + "\n")

custom_model = load_model("Models/eye_model.h5")
custom_loss, custom_acc = custom_model.evaluate(test_data)


# ===========================
# COMPARISON RESULTS
# ===========================
print("\n" + "="*60)
print("EYE MODEL COMPARISON RESULTS")
print("="*60)
print(f"\n1. Custom CNN (Baseline)")
print(f"   - Test Accuracy : {custom_acc * 100:.2f}%")
print(f"   - Test Loss     : {custom_loss:.4f}")
print(f"\n2. MobileNetV2 FROZEN (Feature Extraction)")
print(f"   - Test Accuracy : {frozen_acc * 100:.2f}%")
print(f"   - Test Loss     : {frozen_loss:.4f}")
print(f"   - Training Time : {frozen_train_time:.1f}s")
print(f"\n3. MobileNetV2 FINE-TUNED (Last 30 Layers)")
print(f"   - Test Accuracy : {tuned_acc * 100:.2f}%")
print(f"   - Test Loss     : {tuned_loss:.4f}")
print(f"   - Training Time : {tuned_train_time:.1f}s")
print("\n" + "="*60)
