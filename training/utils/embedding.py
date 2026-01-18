import torch
import numpy as np
from tqdm import tqdm

def get_phobert_embedding(text, model, tokenizer, device, max_length=256):
    """Tạo embedding cho một text đơn lẻ"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding[0]

def get_phobert_embeddings_batch(texts, model, tokenizer, device, batch_size=32, max_length=256):
    """Tạo embeddings cho nhiều texts với batch processing (tận dụng GPU tốt hơn)"""
    model.eval()
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Tạo embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Tạo embeddings trên GPU
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Lấy CLS token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
    
    return np.vstack(all_embeddings)
