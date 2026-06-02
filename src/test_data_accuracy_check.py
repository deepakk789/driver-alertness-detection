import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


eye_model = load_model("Models/eye_model.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    r"D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED\TEST\EYE_TEST",
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# loss, accuracy = eye_model.evaluate(test_data)

# print("Test Accuracy:", accuracy * 100)
# print("Total Test Images:", test_data.samples)

y_pred = eye_model.predict(test_data)
y_pred = (y_pred > 0.5).astype(int)

# True labels
y_true = test_data.classes

# Accuracy
acc = accuracy_score(y_true, y_pred)

print("Accuracy:", acc * 100)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)