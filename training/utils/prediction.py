from .text_preprocess import preprocess_text
from .embedding import get_phobert_embedding
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def predict_sentiment(text, model):
    text_cleaned = preprocess_text(text)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phobert.to(device)

    emb = get_phobert_embedding(text_cleaned, phobert, tokenizer, device)
    pred = model.predict(emb.reshape(1, -1))[0]
    label_names = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}
    return {'text': text, 'sentiment': label_names[pred], 'label_code': pred}
