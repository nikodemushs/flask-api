# app.py  ← only the parts that changed / were added
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os

app = Flask(__name__)
CORS(app)

label_mapping = {0: "Normal", 1: "Fraud / Penipuan", 2: "Promo"}

DEFAULT_MODEL = "80-20/best_nb_tfidf_model_80_20.pkl"   # fallback

_CACHE = {}

def load_bundle(path):
    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file '{path}' not found")
        with open(path, "rb") as f:
            _CACHE[path] = pickle.load(f)
    return _CACHE[path]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        text       = data.get("text", "")
        model_path = data.get("model_path", DEFAULT_MODEL)

        if not text:
            return jsonify({"error": "No text provided"}), 400

        bundle     = load_bundle(model_path)
        model      = bundle["model"]
        vectorizer = bundle["vectorizer"]

        X = vectorizer.transform([text])
        pred_num = int(model.predict(X)[0])
        probs    = model.predict_proba(X)[0]

        return jsonify({
            "ModelPath":   model_path,
            "Prediction":  label_mapping.get(pred_num, "Unknown"),
            "Label":       pred_num,
            "Probabilities": {
                label_mapping.get(i, "Unknown"): f"{p*100:.2f}%"
                for i, p in enumerate(probs)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
