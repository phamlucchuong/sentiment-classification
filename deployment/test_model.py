from predictor import get_sentiment

# Test với các câu rõ ràng là NEGATIVE
negative_texts = [
    "Sản phẩm tệ, chất lượng kém",
    "Rất tệ, không nên mua",
    "Giao hàng chậm, shop tệ",
    "Đồ rất dở, không đáng tiền",
    "Thất vọng, sản phẩm rác",
    "Chất lượng kém, không giống hình",
    "Quá tệ, không bao giờ mua nữa",
    "Dở tệ, lãng phí tiền",
]

# Test với các câu POSITIVE
positive_texts = [
    "Sản phẩm tốt, rất hài lòng",
    "Rất tuyệt vời, đáng tiền",
    "Giao hàng nhanh, shop uy tín",
    "Chất lượng tốt, giống hình",
]

print("=" * 70)
print("TESTING NEGATIVE SENTENCES")
print("=" * 70)
for text in negative_texts:
    result = get_sentiment(text)
    print(f"\nText: {text}")
    print(f"Label: {result['label']} (Confidence: {result['confidence']:.3f})")
    print(f"Probs: NEG={result['probabilities']['NEGATIVE']:.3f}, POS={result['probabilities']['POSITIVE']:.3f}")

print("\n" + "=" * 70)
print("TESTING POSITIVE SENTENCES")
print("=" * 70)
for text in positive_texts:
    result = get_sentiment(text)
    print(f"\nText: {text}")
    print(f"Label: {result['label']} (Confidence: {result['confidence']:.3f})")
    print(f"Probs: NEG={result['probabilities']['NEGATIVE']:.3f}, POS={result['probabilities']['POSITIVE']:.3f}")
