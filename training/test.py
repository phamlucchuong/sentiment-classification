import joblib
import torch
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------
# 1. TẢI CÁC MODEL (giống hệt predictor.py)
# ----------------------------------------------------
print("Đang tải model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải PhoBERT
PHOBERT_MODEL_NAME = "vinai/phobert-base"
phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
phobert_model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
phobert_model.to(device)
phobert_model.eval()

# Tải model Logistic Regression (đường dẫn tới file pkl)
# Hãy chắc chắn đường dẫn này là đúng
MODEL_FILE_PATH = "models/best_sentiment_model.pkl" 
lr_model = joblib.load(MODEL_FILE_PATH)

print("Tải model thành công!")

# ----------------------------------------------------
# 2. HÀM DỰ ĐOÁN (giống hệt predictor.py)
# ----------------------------------------------------
def get_sentiment(text: str) -> str:
    # 1. Tokenize
    inputs = phobert_tokenizer(
        text, 
        add_special_tokens=True, 
        max_length=256, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 2. Lấy embedding
    with torch.no_grad():
        outputs = phobert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :].cpu().numpy()

    # 3. Dự đoán
    prediction = lr_model.predict(cls_embedding)
    
    # 4. Trả về kết quả
    if prediction[0] == 2:
        return "Tích cực"
    elif prediction[0] == 1:
        return "Trung tính"
    else:
        return "Tiêu cực"

# ----------------------------------------------------
# 3. KIỂM THỬ VỚI CÂU CỦA BẠN
# ----------------------------------------------------
if __name__ == "__main__":
    
    my_sentence_1 = "sản phẩm này dùng cũng tạm được"
    my_sentence_2 = "shop nên xem lại thái độ phục vụ"
    my_sentence_3 = "tuyệt vời, không có gì để chê"

    print("--- Bắt đầu kiểm thử ---")
    print(f"Câu: '{my_sentence_1}' -> Dự đoán: {get_sentiment(my_sentence_1)}")
    print(f"Câu: '{my_sentence_2}' -> Dự đoán: {get_sentiment(my_sentence_2)}")
    print(f"Câu: '{my_sentence_3}' -> Dự đoán: {get_sentiment(my_sentence_3)}")