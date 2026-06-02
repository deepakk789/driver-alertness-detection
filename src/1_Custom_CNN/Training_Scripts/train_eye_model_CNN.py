import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen=ImageDataGenerator(rescale=1./255)

train_data=train_datagen.flow_from_directory(
    r"D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED\TRAIN\EYE_TRAIN",
    batch_size=32,
    target_size=(96,96),
    class_mode='binary'
)

val_data=val_datagen.flow_from_directory(
    r"D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED\VALIDATION\EYE_VAL",
    target_size=(96,96),
    batch_size=32,
    class_mode="binary"
)


model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(96,96,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(1,activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

callbacks=[
    EarlyStopping(patience=5,restore_best_weights=True),
    ModelCheckpoint("Models/eye_model.h5",save_best_only=True)
]

history=model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)