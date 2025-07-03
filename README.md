# Emotion Classification API

This project is a text-based emotion classification API using a pre-trained Support Vector Machine (SVM) model. The API is built with FastAPI.

The model was trained on the GoEmotions dataset and can predict one of many emotions from a given text.

## Dependencies

The main dependencies are:
- FastAPI
- Scikit-learn
- Joblib
- Pandas

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run

To start the API server, run the following command from the project's root directory:

```bash
uvicorn scripts.api:app --reload
```

## How to Use

The API has a single endpoint: `/predict`.

### Endpoint: `/predict`

-   **Method:** `POST`
-   **Description:** Predicts the emotion from a given text.
-   **Request Body:** A JSON object with a single key "text".
    ```json
    {
      "text": "I am so happy today!"
    }
    ```
-   **Response:** A JSON object containing the predicted "emotion" and the "confidence" score.
    ```json
    {
      "emotion": "admiration",
      "confidence": 0.99
    }
    ```

### Example Usage with cURL

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I love this new phone, it is amazing!"
}'
```