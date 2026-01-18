"""
Script để train lại model với các tham số tốt hơn để tránh overfitting
"""
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load data
print("Loading data...")
embeddings = np.load('../training/data/processed/embeddings.npy')
labels = np.load('../training/data/processed/labels.npy')

print(f"Total samples: {len(labels)}")
print(f"  - NEGATIVE (0): {np.sum(labels == 0)}")
print(f"  - POSITIVE (2): {np.sum(labels == 2)}")

# Chia train/validation/test với tỷ lệ 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    embeddings, labels, test_size=0.15, random_state=42, stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"\nData split:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(labels)*100:.1f}%)")
print(f"  Val:   {len(X_val)} ({len(X_val)/len(labels)*100:.1f}%)")
print(f"  Test:  {len(X_test)} ({len(X_test)/len(labels)*100:.1f}%)")

# Tính class weight để xử lý imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 2: class_weights[1]}
print(f"\nClass weights: {class_weight_dict}")

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

models = {
    'LogisticRegression (C=0.1)': LogisticRegression(
        C=0.1,  # Regularization mạnh hơn
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'LogisticRegression (C=1.0)': LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'LogisticRegression (C=10.0)': LogisticRegression(
        C=10.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'RandomForest (max_depth=10)': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # Giới hạn depth để tránh overfit
        class_weight='balanced',
        random_state=42
    ),
    'RandomForest (max_depth=20)': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    ),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Đánh giá trên từng tập
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    # Xem chi tiết validation
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    
    # Tính accuracy cho từng class trên validation
    neg_mask = y_val == 0
    pos_mask = y_val == 2
    
    neg_acc = accuracy_score(y_val[neg_mask], val_pred[neg_mask])
    pos_acc = accuracy_score(y_val[pos_mask], val_pred[pos_mask])
    
    results.append({
        'name': name,
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'neg_acc': neg_acc,
        'pos_acc': pos_acc,
        'gap': train_acc - val_acc  # Overfit indicator
    })
    
    print(f"  Train: {train_acc:.4f}")
    print(f"  Val:   {val_acc:.4f} (NEG: {neg_acc:.4f}, POS: {pos_acc:.4f})")
    print(f"  Test:  {test_acc:.4f}")
    print(f"  Gap:   {train_acc - val_acc:.4f}")

# Chọn model tốt nhất dựa trên validation accuracy và gap nhỏ
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"{'Model':<35} {'Train':>7} {'Val':>7} {'Test':>7} {'Gap':>7} {'NEG_Acc':>7} {'POS_Acc':>7}")
print("-"*70)

for r in sorted(results, key=lambda x: (x['val_acc'], -x['gap']), reverse=True):
    print(f"{r['name']:<35} {r['train_acc']:>7.4f} {r['val_acc']:>7.4f} {r['test_acc']:>7.4f} {r['gap']:>7.4f} {r['neg_acc']:>7.4f} {r['pos_acc']:>7.4f}")

# Chọn model tốt nhất (val_acc cao nhất và gap nhỏ)
best_result = max(results, key=lambda x: (x['val_acc'], -x['gap']))
best_model = best_result['model']
best_name = best_result['name']

print(f"\n" + "="*70)
print(f"BEST MODEL: {best_name}")
print("="*70)

# Đánh giá chi tiết trên test set
test_pred = best_model.predict(X_test)
print("\nTest Set Classification Report:")
print(classification_report(y_test, test_pred, target_names=['NEGATIVE', 'POSITIVE']))

print("\nTest Set Confusion Matrix:")
cm = confusion_matrix(y_test, test_pred)
print("              Pred NEG  Pred POS")
print(f"Actual NEG       {cm[0][0]:4d}      {cm[0][1]:4d}")
print(f"Actual POS       {cm[1][0]:4d}      {cm[1][1]:4d}")

# Lưu model
joblib.dump(best_model, 'models/best_sentiment_model_v2.pkl')
print(f"\n✓ Saved improved model to 'models/best_sentiment_model_v2.pkl'")
print(f"\nTo use this model, update MODEL_FILE_PATH in predictor.py to:")
print(f"  MODEL_FILE_PATH = 'models/best_sentiment_model_v2.pkl'")
