import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurations
BASE_DIR = r"D:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM"
TEST_DATA_DIR = os.path.join(BASE_DIR, "DATASET_COMBINED", "TEST")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Presentation_Metrics")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model paths
eye_model_path = os.path.join(MODELS_DIR, "eye_model.h5")
yawn_model_path = os.path.join(MODELS_DIR, "yawn_model.h5")
head_model_path = os.path.join(MODELS_DIR, "head_model_mobilenet_tuned.h5")

# Dataset paths
eye_test_dir = os.path.join(TEST_DATA_DIR, "EYE_TEST")
yawn_test_dir = os.path.join(TEST_DATA_DIR, "YAWN_TEST")
head_test_dir = os.path.join(TEST_DATA_DIR, "HEAD_TEST")

def evaluate_model(model_path, test_dir, model_name, class_names):
    print(f"\n================ Evaluating {model_name} ================")
    
    # Check if model and test dir exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(test_dir):
        print(f"Error: Test dataset not found at {test_dir}")
        return

    # Load Model
    print(f"Loading {model_name}...")
    model = load_model(model_path)
    
    # Load Data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print(f"Loading test data from {test_dir}...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(96, 96),
        batch_size=32,
        class_mode='binary',
        shuffle=False # IMPORTANT for confusion matrix
    )
    
    # Get true labels and predict
    print("Running predictions...")
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate Metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n--- {model_name} Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    # 1. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix\nPrecision: {precision:.2f} | Recall: {recall:.2f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"{model_name}_Confusion_Matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")
    
    # 2. Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, f"{model_name}_ROC_Curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")


# Execute evaluations
evaluate_model(eye_model_path, eye_test_dir, "Eye_Model", ["Closed", "Open"])
evaluate_model(yawn_model_path, yawn_test_dir, "Yawn_Model", ["Not_Yawn", "Yawn"])
evaluate_model(head_model_path, head_test_dir, "Head_Pose_Model", ["Looking_Away", "Looking_Forward"])

print(f"\nAll metrics generation complete! Images saved to {OUTPUT_DIR}")
