import joblib
import numpy as np

# Load model
model = joblib.load('models/best_sentiment_model.pkl')

print("=" * 70)
print("MODEL INFORMATION")
print("=" * 70)
print(f"Model type: {type(model).__name__}")
print(f"Classes: {model.classes_}")
print(f"Number of features: {model.coef_.shape[1] if hasattr(model, 'coef_') else 'N/A'}")

if hasattr(model, 'coef_'):
    print(f"\nCoefficients shape: {model.coef_.shape}")
    print(f"Intercept: {model.intercept_}")
    
    # Kiểm tra phân bố hệ số
    coef_stats = {
        'mean': np.mean(model.coef_),
        'std': np.std(model.coef_),
        'min': np.min(model.coef_),
        'max': np.max(model.coef_),
        'positive_count': np.sum(model.coef_ > 0),
        'negative_count': np.sum(model.coef_ < 0),
    }
    print(f"\nCoefficient statistics:")
    for key, val in coef_stats.items():
        print(f"  {key}: {val}")

# Kiểm tra accuracy trên training data nếu có
try:
    embeddings = np.load('../training/data/processed/embeddings.npy')
    labels = np.load('../training/data/processed/labels.npy')
    
    predictions = model.predict(embeddings)
    accuracy = np.mean(predictions == labels)
    
    print(f"\n" + "=" * 70)
    print(f"TRAINING DATA EVALUATION")
    print("=" * 70)
    print(f"Total samples: {len(labels)}")
    print(f"Training accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    unique_labels = np.unique(labels)
    print(f"\nPrediction distribution:")
    for label in unique_labels:
        pred_count = np.sum(predictions == label)
        true_count = np.sum(labels == label)
        print(f"  Label {label}: predicted={pred_count}, actual={true_count}")
    
    # Chi tiết confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"  Actual→  Pred 0    Pred 2")
    for actual_label in unique_labels:
        mask = labels == actual_label
        pred_0 = np.sum(predictions[mask] == 0)
        pred_2 = np.sum(predictions[mask] == 2)
        print(f"  Label {actual_label}:   {pred_0:6d}    {pred_2:6d}")
        
except Exception as e:
    print(f"\nCannot load training data: {e}")
