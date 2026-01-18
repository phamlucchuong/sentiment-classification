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
def get_sentiment(text: str) -> tuple:
    """
    Trả về (label_số, label_text, probability) để dễ theo dõi
    """
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
    
    # Lấy xác suất nếu model hỗ trợ
    try:
        probabilities = lr_model.predict_proba(cls_embedding)[0]
        confidence = max(probabilities)
    except:
        confidence = None
    
    # 4. Trả về kết quả với label số và text
    label_num = int(prediction[0])
    if label_num == 1:
        label_text = "Tích cực (Positive)"
    elif label_num == 0:
        label_text = "Tiêu cực (Negative)"
    else:
        label_text = f"Không xác định (Label={label_num})"
    
    return label_num, label_text, confidence

# ----------------------------------------------------
# 3. KIỂM THỬ TOÀN DIỆN
# ----------------------------------------------------
if __name__ == "__main__":
    
    # Test cases đa dạng để kiểm tra model dự đoán được mấy nhãn
    test_cases = [
        # Positive rõ ràng
        ("Món ăn rất ngon, phục vụ tuyệt vời, tôi sẽ quay lại", "Positive"),
        ("Tuyệt vời, không có gì để chê", "Positive"),
        ("Sản phẩm chất lượng, giá cả hợp lý, rất hài lòng", "Positive"),
        ("Đồ ăn tươi ngon, nhân viên thân thiện, không gian đẹp", "Positive"),
        
        # Negative rõ ràng
        ("Quán này tệ lắm, đồ ăn dở, không bao giờ quay lại", "Negative"),
        ("Shop nên xem lại thái độ phục vụ, rất tệ", "Negative"),
        ("Món ăn cực tệ, chất lượng kém", "Negative"),
        ("Thất vọng, lãng phí tiền, không nên đến", "Negative"),
        ("Đồ ăn nguội lạnh, chờ lâu, nhân viên thờ ơ", "Negative"),
        
        # Trung tính /애매한 trường hợp
        ("Sản phẩm này dùng cũng tạm được", "Neutral/Unclear"),
        ("Bình thường, không có gì đặc biệt", "Neutral/Unclear"),
        ("Giá hơi cao nhưng cũng được", "Neutral/Unclear"),
    ]
    
    print("=" * 80)
    print("🧪 KIỂM THỬ MODEL - DỰ ĐOÁN ĐƯỢC BAO NHIÊU NHÃN?")
    print("=" * 80)
    print(f"📊 Model: {MODEL_FILE_PATH}")
    print(f"🖥️  Device: {device}")
    print("=" * 80)
    
    # Thống kê kết quả
    predictions_count = {}
    results = []
    
    print("\n📝 KẾT QUẢ DỰ ĐOÁN CHI TIẾT:")
    print("-" * 80)
    
    for i, (text, expected_type) in enumerate(test_cases, 1):
        label_num, label_text, confidence = get_sentiment(text)
        
        # Đếm các label
        predictions_count[label_num] = predictions_count.get(label_num, 0) + 1
        
        # Hiển thị kết quả
        conf_str = f"{confidence:.2%}" if confidence else "N/A"
        print(f"\n[Test {i:2d}] Loại mong đợi: {expected_type}")
        print(f"  📄 Input: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        print(f"  🎯 Dự đoán: Label={label_num} | {label_text} | Confidence={conf_str}")
        
        results.append({
            'text': text,
            'expected': expected_type,
            'label': label_num,
            'label_text': label_text,
            'confidence': confidence
        })
    
    # Thống kê tổng quan
    print("\n" + "=" * 80)
    print("📊 THỐNG KÊ TỔNG QUAN:")
    print("=" * 80)
    print(f"✅ Tổng số test cases: {len(test_cases)}")
    print(f"🏷️  Số nhãn duy nhất được dự đoán: {len(predictions_count)}")
    print(f"\n📈 Phân bố dự đoán:")
    for label, count in sorted(predictions_count.items()):
        label_name = "Negative (0)" if label == 0 else "Positive (1)" if label == 1 else f"Unknown ({label})"
        percentage = count / len(test_cases) * 100
        print(f"   - {label_name}: {count} cases ({percentage:.1f}%)")
    
    # Kiểm tra xem model có dự đoán cả 2 nhãn không
    print("\n" + "=" * 80)
    if len(predictions_count) >= 2:
        print("✅ KẾT LUẬN: Model DỰ ĐOÁN ĐƯỢC NHIỀU NHÃN (không bị overfitting hoàn toàn)")
        if 0 in predictions_count and 1 in predictions_count:
            print("✅ Model dự đoán được cả 2 nhãn: Negative (0) và Positive (1)")
    else:
        print("⚠️  CẢNH BÁO: Model CHỈ DỰ ĐOÁN MỘT NHÃN DUY NHẤT!")
        print("   → Có thể model vẫn đang bị overfitting nghiêm trọng")
        print("   → Cần huấn luyện lại với class_weight='balanced'")
    print("=" * 80)