import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import joblib

# Load model và embedding từ training
model = joblib.load('models/best_sentiment_model.pkl')
train_embeddings = np.load('../training/data/processed/embeddings.npy')
train_labels = np.load('../training/data/processed/labels.npy')

print("=" * 70)
print("TRAINING EMBEDDINGS")
print("=" * 70)
print(f"Shape: {train_embeddings.shape}")
print(f"Mean: {train_embeddings.mean():.6f}")
print(f"Std: {train_embeddings.std():.6f}")
print(f"Min: {train_embeddings.min():.6f}")
print(f"Max: {train_embeddings.max():.6f}")

# Tạo embedding mới với cùng text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert.to(device)
phobert.eval()

# Test với 1 câu đơn giản
test_text = "sản phẩm này tốt"

inputs = tokenizer(
    test_text,
    add_special_tokens=True,
    max_length=256,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

with torch.no_grad():
    outputs = phobert(input_ids=input_ids, attention_mask=attention_mask)
    new_embedding = outputs[0][:, 0, :].cpu().numpy()

print(f"\n" + "=" * 70)
print("NEW EMBEDDING (from predictor)")
print("=" * 70)
print(f"Shape: {new_embedding.shape}")
print(f"Mean: {new_embedding.mean():.6f}")
print(f"Std: {new_embedding.std():.6f}")
print(f"Min: {new_embedding.min():.6f}")
print(f"Max: {new_embedding.max():.6f}")

# Test trên một mẫu từ training
print(f"\n" + "=" * 70)
print("COMPARE ON TRAINING SAMPLE")
print("=" * 70)

# Lấy 1 mẫu negative và 1 mẫu positive từ training
neg_idx = np.where(train_labels == 0)[0][0]
pos_idx = np.where(train_labels == 2)[0][0]

neg_emb_train = train_embeddings[neg_idx:neg_idx+1]
pos_emb_train = train_embeddings[pos_idx:pos_idx+1]

# Dự đoán với embedding từ training
neg_pred_train = model.predict(neg_emb_train)
neg_proba_train = model.predict_proba(neg_emb_train)

pos_pred_train = model.predict(pos_emb_train)
pos_proba_train = model.predict_proba(pos_emb_train)

print(f"\nNegative sample (label=0):")
print(f"  Prediction: {neg_pred_train[0]}")
print(f"  Probabilities: NEG={neg_proba_train[0][0]:.3f}, POS={neg_proba_train[0][1]:.3f}")

print(f"\nPositive sample (label=2):")
print(f"  Prediction: {pos_pred_train[0]}")
print(f"  Probabilities: NEG={pos_proba_train[0][0]:.3f}, POS={pos_proba_train[0][1]:.3f}")

# Kiểm tra distribution của probabilities
print(f"\n" + "=" * 70)
print("PROBABILITY DISTRIBUTION ON ALL TRAINING DATA")
print("=" * 70)

all_probas = model.predict_proba(train_embeddings)
print(f"Average probability for NEGATIVE class: {all_probas[:, 0].mean():.3f}")
print(f"Average probability for POSITIVE class: {all_probas[:, 1].mean():.3f}")

print(f"\nFor actual NEGATIVE samples:")
neg_mask = train_labels == 0
print(f"  Avg prob NEG: {all_probas[neg_mask, 0].mean():.3f}")
print(f"  Avg prob POS: {all_probas[neg_mask, 1].mean():.3f}")

print(f"\nFor actual POSITIVE samples:")
pos_mask = train_labels == 2
print(f"  Avg prob NEG: {all_probas[pos_mask, 0].mean():.3f}")
print(f"  Avg prob POS: {all_probas[pos_mask, 1].mean():.3f}")
