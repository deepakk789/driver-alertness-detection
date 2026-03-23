import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
#data path
data_dir="D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED"

#data generator
datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data=datagen.flow_from_directory(
    data_dir,
    target_size=(96,96),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data=datagen.flow_from_directory(
    data_dir,
    target_size=(96,96),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

'''making CNN model with the convolution layer(filter map) 
then maxpooling while doing this (also taking relu function (that is max(0,x))
then flattening into the 1D dataset and 
then the fully connected layer comes with the neural networks 
at the end do sigmoid to classify output 
'''

model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(96,96,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1,activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks=[
    EarlyStopping(patience=5,restore_best_weights=True),
    ModelCheckpoint("Models/yawn_model.h5",save_best_only=True)
]

#training the model
history=model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks
)