from fastapi import FastAPI
from pydantic import BaseModel
# Import hàm chính từ file bạn vừa tạo
from predictor import get_sentiment 

app = FastAPI()

# Định nghĩa cấu trúc data đầu vào
class TextInput(BaseModel):
    text: str

# Tạo endpoint /predict
@app.post("/predict")
def predict_sentiment(item: TextInput):
    # Chỉ cần gọi hàm của bạn!
    sentiment = get_sentiment(item.text) 
    
    return {"original_text": item.text, "sentiment": sentiment}