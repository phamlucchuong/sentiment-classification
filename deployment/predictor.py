import joblib  # Hoặc import pickle nếu bạn dùng pickle
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# ======================================================================================
# BƯỚC 1: Tải (load) model một lần khi API khởi động
# ======================================================================================

# Chọn thiết bị (GPU nếu có, không thì CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading models to {device}...")

try:
    # 1. Tải PhoBERT Tokenizer và Model
    # Sử dụng "vinai/phobert-base-v2" nếu bạn dùng version 2
    PHOBERT_MODEL_NAME = "vinai/phobert-base" 
    
    # Tải tokenizer
    phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    
    # Tải model PhoBERT
    phobert_model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
    phobert_model.to(device) # Chuyển model sang device (GPU/CPU)
    phobert_model.eval() # Chuyển sang chế độ đánh giá (quan trọng!)

    # 2. Tải mô hình Logistic Regression đã lưu
    # ---!!! CHỈNH SỬA TÊN FILE CỦA BẠN TẠI ĐÂY !!!---
    MODEL_FILE_PATH = "models/best_sentiment_model.pkl" 
    
    lr_model = joblib.load(MODEL_FILE_PATH)

    print("Models loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")
    # Bạn có thể xử lý lỗi ở đây, ví dụ: thoát ứng dụng
    lr_model = None
    phobert_model = None
    phobert_tokenizer = None

# ======================================================================================
# BƯỚC 2: Định nghĩa hàm xử lý và dự đoán
# ======================================================================================

def get_sentiment(text: str) -> dict:
    """
    Hàm này nhận một chuỗi văn bản thô, 
    trích xuất embedding từ PhoBERT,
    và dự đoán cảm xúc bằng mô hình Logistic Regression.
    Trả về dictionary chứa nhãn dự đoán và xác suất cho cả 2 nhãn.
    """
    
    if lr_model is None or phobert_model is None or phobert_tokenizer is None:
        return {"error": "Mô hình chưa được tải"}

    try:
        # 1. Tokenize văn bản
        # max_length: Chiều dài tối đa của câu (PhoBERT là 256)
        # padding='max_length': Đệm cho các câu ngắn hơn
        # truncation=True: Cắt các câu dài hơn
        # return_tensors='pt': Trả về PyTorch tensors
        inputs = phobert_tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=256, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        # Chuyển input tensors sang device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 2. Lấy embedding từ PhoBERT
        # torch.no_grad() để không tính toán gradient (tiết kiệm bộ nhớ và tăng tốc)
        with torch.no_grad():
            outputs = phobert_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Lấy embedding của token [CLS] (token đầu tiên)
            # Đây là vector đại diện cho toàn bộ câu
            # outputs[0] là last_hidden_state
            cls_embedding = outputs[0][:, 0, :].cpu().numpy()

        # 3. Dự đoán bằng Logistic Regression
        # cls_embedding đang có shape (1, 768), sẵn sàng cho mô hình sklearn
        prediction = lr_model.predict(cls_embedding)
        probabilities = lr_model.predict_proba(cls_embedding)
        
        # 4. Trả về kết quả dạng dictionary
        # Lấy danh sách các nhãn từ model (ví dụ: [0, 2])
        classes = lr_model.classes_
        
        # Ánh xạ nhãn số sang nhãn text
        label_map = {
            0: "NEGATIVE",
            2: "POSITIVE"
        }
        
        predicted_label = int(prediction[0])
        
        # Tìm index của nhãn dự đoán trong mảng classes
        # Ví dụ: nếu prediction[0]=2 và classes=[0,2], thì index=1
        predicted_index = np.where(classes == predicted_label)[0][0]
        
        # Tạo dictionary probabilities dựa trên classes thực tế
        probs_dict = {}
        for idx, class_label in enumerate(classes):
            probs_dict[label_map.get(class_label, f"CLASS_{class_label}")] = float(probabilities[0][idx])
        
        result = {
            "label": label_map.get(predicted_label, f"CLASS_{predicted_label}"),
            "confidence": float(probabilities[0][predicted_index]),
            "probabilities": probs_dict
        }
        
        return result

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "Lỗi trong quá trình dự đoán"}

# ======================================================================================
# BƯỚC 3: Thử nghiệm (chạy trực tiếp file này)
# ======================================================================================

if __name__ == "__main__":
    # Bạn có thể chạy 'python predictor.py' để kiểm tra nhanh
    
    test_text_1 = "sản phẩm này tốt thật sự"
    test_text_2 = "giao hàng quá chậm, shop làm ăn chán đời"
    
    print(f"Câu: '{test_text_1}'")
    print(f"Kết quả: {get_sentiment(test_text_1)}\n")
    
    print(f"Câu: '{test_text_2}'")
    print(f"Kết quả: {get_sentiment(test_text_2)}")