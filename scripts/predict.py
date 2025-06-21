import joblib
import sys

def predict_emotion(text, model):
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    return pred, max(prob)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'Your text here'")
        sys.exit(1)

    text = sys.argv[1]
    model = joblib.load("../models/svm_goemotions.pkl")

    emotion, confidence = predict_emotion(text, model)
    print(f"Predicted emotion: {emotion}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
